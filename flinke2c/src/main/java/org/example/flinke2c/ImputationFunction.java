package org.example.flinke2c;

import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.annotation.FunctionHint;
import org.apache.flink.table.functions.ScalarFunction;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

/**
 * KNN price imputation over the last observed bids (one history per subtask, not per key).
 * History is not checkpointed.
 */
public class ImputationFunction extends ScalarFunction {
    private static final int HISTORY_SIZE = 5_000; // memory

    // How many rows are scanned per missing record (CPU)
    private static final int SEARCH_LIMIT = 512;

    private static final int K = 10; // CPU

    private static final double EPS = 1e-6;

    // feature weights
    private static final double W_BIDDER = 0.25;
    private static final double W_TIME = 1.0;
    private static final double W_STR = 0.25;

    // defaults
    private static final BigDecimal DEFAULT_PRICE = new BigDecimal("0.000");
    private static final long DEFAULT_LONG = 0L;
    private static final String DEFAULT_CHANNEL = "unknown";
    private static final String DEFAULT_STRING = "";
    private static final LocalDateTime DEFAULT_TIMESTAMP = LocalDateTime.of(1970, 1, 1, 0, 0);

    // Global ring (per JVM/subtask)
    private static final BoundedRing HISTORY = new BoundedRing(HISTORY_SIZE);

    @FunctionHint(
            input = {
                    @DataTypeHint("DECIMAL(23,3)"),
                    @DataTypeHint("BIGINT"),
                    @DataTypeHint("BIGINT"),
                    @DataTypeHint("STRING"),
                    @DataTypeHint("STRING"),
                    @DataTypeHint("TIMESTAMP(3)"),
                    @DataTypeHint("STRING")
            },
            output = @DataTypeHint(bridgedTo = ImputedBid.class)
    )
    public ImputedBid eval(
            BigDecimal price,
            Long auction,
            Long bidder,
            String channel,
            String url,
            LocalDateTime dateTime,
            String extra) {

        long a = auction == null ? DEFAULT_LONG : auction;
        long b = bidder == null ? DEFAULT_LONG : bidder;
        String ch = isBlank(channel) ? DEFAULT_CHANNEL : channel;
        String u = isBlank(url) ? DEFAULT_STRING : url;
        LocalDateTime dt = dateTime == null ? DEFAULT_TIMESTAMP : dateTime;
        String ex = isBlank(extra) ? DEFAULT_STRING : extra;

        ImputedBid out = new ImputedBid(price, a, b, ch, u, dt, ex);

        // Build obs for similarity (no arrays, no heavy allocations)
        Obs obs = Obs.from(out);

        if (price != null) {
            // only observed prices go into history
            obs.hasPrice = true;
            obs.priceDouble = price.doubleValue();
            HISTORY.add(obs);
            return out;
        }

        // impute using KNN over a recent slice
        double imputed = knnImputePrice(obs);
        if (Double.isNaN(imputed)) {
            out.price = DEFAULT_PRICE;
        } else {
            out.price = BigDecimal.valueOf(imputed).setScale(3, RoundingMode.HALF_UP);
        }

        return out;
    }

    private static double knnImputePrice(Obs target) {
        // Copy last SEARCH_LIMIT obs references quickly under lock, compute outside lock.
        Obs[] snap = HISTORY.snapshotLast(Math.min(SEARCH_LIMIT, HISTORY_SIZE));
        if (snap.length == 0) return Double.NaN;

        // top-K buffers (no sorting, no list allocations)
        double[] bestDist = new double[K];
        double[] bestPrice = new double[K];
        int found = 0;

        for (Obs o : snap) {
            if (o == null || !o.hasPrice) continue;

            double dist = distance(target, o);

            // insert into top-K (keep smallest distances)
            if (found < K) {
                bestDist[found] = dist;
                bestPrice[found] = o.priceDouble;
                found++;
            } else {
                int worstIdx = 0;
                double worst = bestDist[0];
                for (int j = 1; j < K; j++) {
                    if (bestDist[j] > worst) {
                        worst = bestDist[j];
                        worstIdx = j;
                    }
                }
                if (dist < worst) {
                    bestDist[worstIdx] = dist;
                    bestPrice[worstIdx] = o.priceDouble;
                }
            }
        }

        if (found == 0) return Double.NaN;

        // weighted average
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < found; i++) {
            double w = 1.0 / (bestDist[i] + EPS);
            num += bestPrice[i] * w;
            den += w;
        }
        return den == 0.0 ? Double.NaN : (num / den);
    }

    private static double distance(Obs t, Obs o) {
        double s = 0.0;

        // bidder: 0/1 mismatch
        s += W_BIDDER * (t.bidderId == o.bidderId ? 0.0 : 1.0);

        // time: squared diff scaled down
        double dt = (t.tsSeconds - o.tsSeconds);
        s += W_TIME * (dt * dt) * 1e-8;

        // strings: hashed equality signals
        s += W_STR * (t.channelHash == o.channelHash ? 0.0 : 1.0);
        s += W_STR * (t.urlHash == o.urlHash ? 0.0 : 1.0);
        s += W_STR * (t.extraHash == o.extraHash ? 0.0 : 1.0);

        return s;
    }

    private static final class BoundedRing {
        final int capacity;
        final Obs[] buffer;
        final Object lock = new Object();
        int start = 0;
        int size = 0;

        BoundedRing(int capacity) {
            this.capacity = capacity;
            this.buffer = new Obs[capacity];
        }

        void add(Obs obs) {
            synchronized (lock) {
                if (size < capacity) {
                    buffer[(start + size) % capacity] = obs;
                    size++;
                } else {
                    buffer[start] = obs;
                    start = (start + 1) % capacity;
                }
            }
        }

        Obs[] snapshotLast(int limit) {
            synchronized (lock) {
                int n = Math.min(size, limit);
                Obs[] out = new Obs[n];
                for (int i = 0; i < n; i++) {
                    // last elements first
                    int idx = (start + size - 1 - i + capacity) % capacity;
                    out[i] = buffer[idx];
                }
                return out;
            }
        }
    }

    public static final class Obs implements Serializable {
        public boolean hasPrice;
        public double priceDouble;

        public long bidderId;
        public double tsSeconds;
        public int channelHash;
        public int urlHash;
        public int extraHash;

        static Obs from(ImputedBid bid) {
            Obs o = new Obs();
            o.bidderId = bid.bidder;
            o.tsSeconds = bid.dateTime == null
                    ? 0.0
                    : bid.dateTime.toInstant(ZoneOffset.UTC).toEpochMilli() / 1000.0;
            o.channelHash = hashOrZero(bid.channel);
            o.urlHash = hashOrZero(bid.url);
            o.extraHash = hashOrZero(bid.extra);
            return o;
        }

        private static int hashOrZero(String s) {
            if (s == null || s.isBlank()) return 0;
            return murmurLikeHash(s);
        }

        private static int murmurLikeHash(String s) {
            byte[] data = s.getBytes(StandardCharsets.UTF_8);
            int h = 0x9747b28c;
            for (byte b : data) {
                h ^= b;
                h *= 0x5bd1e995;
                h ^= (h >>> 15);
            }
            return h;
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    public static class ImputedBid {
        @DataTypeHint("DECIMAL(23,3)")
        public BigDecimal price;
        public long auction;
        public long bidder;
        public String channel;
        public String url;
        @DataTypeHint("TIMESTAMP(3)")
        public LocalDateTime dateTime;
        public String extra;

        public ImputedBid() {}

        public ImputedBid(BigDecimal price, long auction, long bidder, String channel, String url, LocalDateTime dateTime, String extra) {
            this.price = price;
            this.auction = auction;
            this.bidder = bidder;
            this.channel = channel;
            this.url = url;
            this.dateTime = dateTime;
            this.extra = extra;
        }
    }
}
