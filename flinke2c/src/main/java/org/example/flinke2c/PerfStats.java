package org.example.flinke2c;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Writer;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Per-stage wall-clock accumulator for the packed direct-CUDA GPU functions,
 * so a real run answers "where did the time go" instead of guessing from
 * throughput alone.
 *
 * <p>Stages are named at construction time (fixed set, so every CSV row
 * written by a given instance has the same columns - required for a valid
 * CSV when rows from different runs are appended to the same file over
 * time). {@link #record} and the {@link #start} scope are backed by
 * {@link LongAdder}, so recording from multiple threads (the Flink thread,
 * the submitter thread, the completer thread) never blocks on a shared lock.
 *
 * <p>{@link #appendCsv} writes one summary row per {@code open()}..{@code
 * close()} lifecycle - "new connection" in the caller's terms, since this is
 * the closest local-GPU equivalent to the RDMA path's per-session
 * connection - with the aggregate and average for every stage, plus
 * whatever extra context columns the caller supplies (row/batch counts,
 * the config in effect, ...). Best-effort: a write failure never fails the
 * job (this is diagnostics, not job output) but is logged at WARN on the
 * TaskManager - check there, not just the client, if a row never shows up;
 * on a cluster the file is written on whichever TaskManager the operator
 * instance is running on, not necessarily the machine running the SQL
 * client. The path actually used is also logged at INFO on the first
 * successful write, resolved to an absolute path, so there's no ambiguity
 * about where to look.
 */
final class PerfStats {
    private static final Logger LOG = LoggerFactory.getLogger(PerfStats.class);
    // Resolved once per JVM and reused: this class only ever runs inside
    // whichever TaskManager the operator instance was scheduled onto - the
    // GPU it drives is local to that machine - which is not necessarily,
    // and on a real cluster usually isn't, the machine running the SQL
    // client or checking the path afterward. Logged alongside the path so
    // "which host is this file even on" has a direct answer in the log
    // line instead of needing a trip through the Flink UI's task list.
    private static final String LOCAL_HOST = resolveLocalHost();

    private final String[] stages;
    private final Map<String, LongAdder> nanosByStage = new ConcurrentHashMap<>();
    private final Map<String, LongAdder> countByStage = new ConcurrentHashMap<>();
    private final AtomicBoolean announced = new AtomicBoolean(false);

    PerfStats(String... stages) {
        this.stages = stages.clone();
        for (String stage : this.stages) {
            nanosByStage.put(stage, new LongAdder());
            countByStage.put(stage, new LongAdder());
        }
    }

    /** Records one occurrence of {@code stage} taking {@code elapsedNanos}. No allocation. */
    void record(String stage, long elapsedNanos) {
        nanosByStage.get(stage).add(elapsedNanos);
        countByStage.get(stage).increment();
    }

    /** Convenience try-with-resources scope for stages where one small allocation per call is fine. */
    Scope start(String stage) {
        return new Scope(stage);
    }

    final class Scope implements AutoCloseable {
        private final String stage;
        private final long startNanos;

        private Scope(String stage) {
            this.stage = stage;
            this.startNanos = System.nanoTime();
        }

        @Override
        public void close() {
            record(stage, System.nanoTime() - startNanos);
        }
    }

    /**
     * Appends one row to {@code csvPath}: {@code timestamp,host,label,wallMillis},
     * then every entry of {@code extraColumns} in iteration order, then
     * {@code <stage>TotalMillis,<stage>AvgMicros,<stage>Count} for each
     * configured stage. Writes a header line first if the file doesn't
     * already exist. {@code extraColumns} must use the same keys, in the
     * same order, on every call for a given file - the header is only
     * written once. {@code host} is always this JVM's own hostname, mainly
     * useful when rows from several TaskManagers end up copied into one
     * file (e.g. a shared/NFS-mounted csvPath, or parallelism > 1) and it's
     * otherwise not obvious which row came from where.
     */
    synchronized void appendCsv(Path csvPath, String label, long wallMillis, Map<String, String> extraColumns) {
        try {
            boolean writeHeader = !Files.exists(csvPath);
            StringBuilder header = new StringBuilder("timestamp,host,label,wallMillis");
            StringBuilder row = new StringBuilder();
            row.append(Instant.now()).append(',').append(csvSafe(LOCAL_HOST))
                    .append(',').append(csvSafe(label)).append(',').append(wallMillis);
            for (Map.Entry<String, String> extra : extraColumns.entrySet()) {
                header.append(',').append(extra.getKey());
                row.append(',').append(csvSafe(extra.getValue()));
            }
            for (String stage : stages) {
                long nanos = nanosByStage.get(stage).sum();
                long count = countByStage.get(stage).sum();
                double totalMillis = nanos / 1_000_000.0;
                double avgMicros = count == 0 ? 0.0 : (nanos / 1000.0) / count;
                header.append(',').append(stage).append("TotalMillis,")
                        .append(stage).append("AvgMicros,")
                        .append(stage).append("Count");
                row.append(',').append(String.format("%.3f", totalMillis))
                        .append(',').append(String.format("%.3f", avgMicros))
                        .append(',').append(count);
            }
            // Parent directories may not exist yet on a fresh TaskManager.
            if (csvPath.getParent() != null) {
                Files.createDirectories(csvPath.getParent());
            }
            try (Writer writer = Files.newBufferedWriter(csvPath, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                if (writeHeader) {
                    writer.write(header.toString());
                    writer.write('\n');
                }
                writer.write(row.toString());
                writer.write('\n');
            }
            if (announced.compareAndSet(false, true)) {
                LOG.info("PerfStats on {} writing '{}' rows to {}", LOCAL_HOST, label, csvPath.toAbsolutePath());
            }
        } catch (IOException | RuntimeException e) {
            // Diagnostics only; never fail the job over a logging problem.
            // Logged (not swallowed silently) so a permission/path mistake
            // shows up on the TaskManager instead of just "no file appeared".
            LOG.warn("PerfStats on {} failed to append to {}", LOCAL_HOST, csvPath.toAbsolutePath(), e);
        }
    }

    /** Builds the extraColumns map with the header/value pairs in insertion order. */
    static Map<String, String> columns() {
        return new LinkedHashMap<>();
    }

    private static String csvSafe(String value) {
        if (value == null) return "";
        return value.indexOf(',') < 0 && value.indexOf('"') < 0 && value.indexOf('\n') < 0
                ? value
                : '"' + value.replace("\"", "\"\"") + '"';
    }

    private static String resolveLocalHost() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (UnknownHostException e) {
            return "unknown-host";
        }
    }
}
