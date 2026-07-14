package org.apache.flink.table.runtime.functions.table.externalruntime;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Minimal PRE-side JNI/bootstrap smoke test. */
public final class RdmaPreSmokeTest {
    private RdmaPreSmokeTest() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 6) {
            System.err.println("usage: RdmaPreSmokeTest <host> <port> [device] [ibPort] [gidIndex] [rows]");
            System.exit(2);
        }
        String host = args[0];
        int port = Integer.parseInt(args[1]);
        String device = args.length > 2 ? args[2] : "";
        int ibPort = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int gidIndex = args.length > 4 ? Integer.parseInt(args[4]) : 3;
        int rows = args.length > 5 ? Integer.parseInt(args[5]) : 128;

        long handle = RustRdmaNative.open(host, port, device, ibPort, gidIndex, "pre",
                "{\"function\":\"INCREMENT\",\"field_index\":2,\"fields\":[\"INT64\",\"INT64\",\"DECIMAL_BYTES\",\"TIMESTAMP_MILLIS\",\"BYTES\",\"INT64\"]}");
        if (handle == 0) {
            throw new IOException("PRE JNI open returned a null handle");
        }
        try {
            for (int row = 0; row < rows; row++) {
                RustRdmaNative.writeSlot(handle, row(row));
                if ((row + 1) % 64 == 0) {
                    RustRdmaNative.publish(handle, 64);
                }
            }
            int remainder = rows % 64;
            if (remainder != 0) {
                RustRdmaNative.publish(handle, remainder);
            }
            System.out.println("PRE smoke test published " + rows + " rows");
        } finally {
            RustRdmaNative.close(handle);
        }
    }

    private static byte[] row(int rowId) {
        byte[] price = decimalBytes(rowId == 0 ? 0 : 12_345L + rowId);
        byte[] extra = ("smoke-" + rowId).getBytes(StandardCharsets.UTF_8);
        ByteBuffer payload = ByteBuffer.allocate(
                        4 + 8 + 1 + 8 + 8 + 4 + price.length + 8 + 4 + extra.length + 8)
                .order(ByteOrder.BIG_ENDIAN);
        payload.putInt(0);       // op
        payload.putLong(rowId);  // row_id
        payload.put((byte) 0);   // null bitmap: all fields non-null
        payload.putLong(1_000L + rowId); // auction
        payload.putLong(2_000L + rowId); // bidder
        payload.putInt(price.length);    // DECIMAL(23, 3) unscaled byte length
        payload.put(price);
        payload.putLong(1_700_000_000_000L + rowId); // dateTime
        payload.putInt(extra.length);
        payload.put(extra);
        payload.putLong(9_000L + rowId); // latency_ts
        byte[] payloadBytes = payload.array();
        ByteBuffer frame = ByteBuffer.allocate(4 + payloadBytes.length).order(ByteOrder.BIG_ENDIAN);
        frame.putInt(payloadBytes.length);
        frame.put(payloadBytes);
        return frame.array();
    }

    private static byte[] decimalBytes(long unscaled) {
        byte[] bytes = BigInteger.valueOf(unscaled).toByteArray();
        return bytes.length == 0 ? new byte[] {0} : bytes;
    }
}
