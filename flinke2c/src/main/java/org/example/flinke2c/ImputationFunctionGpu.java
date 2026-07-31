package org.example.flinke2c;

import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.annotation.FunctionHint;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.ScalarFunction;

import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

/**
 * Direct CUDA/JNI version of {@link ImputationFunction}.
 *
 * <p>Each {@code eval} call synchronously copies one compact observation to the
 * GPU, launches the standalone imputation kernel, and copies one double back.
 * This intentionally bypasses the RDMA runtime so its measurements can be
 * compared with the RDMA implementation.
 *
 * <p>The history is local to this UDF instance and is not checkpointed.
 */
public class ImputationFunctionGpu extends ScalarFunction {
    private static final BigDecimal DEFAULT_PRICE = new BigDecimal("0.000");
    private static final long DEFAULT_LONG = 0L;
    private static final String DEFAULT_CHANNEL = "unknown";
    private static final String DEFAULT_STRING = "";
    private static final LocalDateTime DEFAULT_TIMESTAMP =
            LocalDateTime.of(1970, 1, 1, 0, 0);

    private final int requestedCudaDevice;
    private transient long nativeHandle;

    /**
     * Uses {@code -Dflinke2c.imputation.gpu.device=N}, or CUDA device zero when
     * the property is absent.
     */
    public ImputationFunctionGpu() {
        this(-1);
    }

    /** Uses the given CUDA device index (after CUDA_VISIBLE_DEVICES filtering). */
    public ImputationFunctionGpu(int cudaDevice) {
        if (cudaDevice < -1) {
            throw new IllegalArgumentException("cudaDevice must be -1 or non-negative");
        }
        this.requestedCudaDevice = cudaDevice;
    }

    @Override
    public synchronized void open(FunctionContext context) {
        ensureNativeHandle();
    }

    @Override
    public synchronized void close() {
        if (nativeHandle != 0L) {
            ImputationGpuNative.destroy(nativeHandle);
            nativeHandle = 0L;
        }
    }

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
            output = @DataTypeHint(bridgedTo = ImputationFunction.ImputedBid.class)
    )
    public synchronized ImputationFunction.ImputedBid eval(
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

        ImputationFunction.ImputedBid out =
                new ImputationFunction.ImputedBid(price, a, b, ch, u, dt, ex);

        ensureNativeHandle();
        double gpuPrice = ImputationGpuNative.process(
                nativeHandle,
                price != null,
                price == null ? 0.0 : price.doubleValue(),
                b,
                dt.toInstant(ZoneOffset.UTC).toEpochMilli() / 1000.0,
                hashOrZero(ch),
                hashOrZero(u),
                hashOrZero(ex));

        if (price == null) {
            out.price = Double.isNaN(gpuPrice)
                    ? DEFAULT_PRICE
                    : BigDecimal.valueOf(gpuPrice).setScale(3, RoundingMode.HALF_UP);
        }
        return out;
    }

    private void ensureNativeHandle() {
        if (nativeHandle == 0L) {
            int device = requestedCudaDevice >= 0
                    ? requestedCudaDevice
                    : Integer.getInteger("flinke2c.imputation.gpu.device", 0);
            nativeHandle = ImputationGpuNative.create(device);
            if (nativeHandle == 0L) {
                throw new IllegalStateException(
                        "CUDA imputation context creation returned a null handle");
            }
        }
    }

    private static int hashOrZero(String value) {
        if (value == null || value.isBlank()) {
            return 0;
        }
        byte[] data = value.getBytes(StandardCharsets.UTF_8);
        int hash = 0x9747b28c;
        for (byte b : data) {
            hash ^= b;
            hash *= 0x5bd1e995;
            hash ^= (hash >>> 15);
        }
        return hash;
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}

/** Package-private JNI entry point kept in this file with the GPU UDF. */
final class ImputationGpuNative {
    static {
        String absoluteLibrary =
                System.getProperty("flinke2c.imputation.gpu.library", "").trim();
        if (absoluteLibrary.isEmpty()) {
            System.loadLibrary("flinke2c_imputation_gpu");
        } else {
            File file = new File(absoluteLibrary);
            if (!file.isAbsolute()) {
                throw new IllegalArgumentException(
                        "flinke2c.imputation.gpu.library must be an absolute path");
            }
            System.load(file.getAbsolutePath());
        }
    }

    private ImputationGpuNative() {}

    static native long create(int cudaDevice);

    static native double process(
            long handle,
            boolean hasPrice,
            double price,
            long bidder,
            double timestampSeconds,
            int channelHash,
            int urlHash,
            int extraHash);

    static native void destroy(long handle);
}
