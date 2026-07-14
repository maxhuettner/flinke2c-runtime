package org.apache.flink.table.runtime.functions.table.externalruntime;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Minimal POST-side JNI/bootstrap smoke test. */
public final class RdmaPostSmokeTest {
    private RdmaPostSmokeTest() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 6) {
            System.err.println("usage: RdmaPostSmokeTest <host> <port> [device] [ibPort] [gidIndex] [rows]");
            System.exit(2);
        }
        String host = args[0];
        int port = Integer.parseInt(args[1]);
        String device = args.length > 2 ? args[2] : "";
        int ibPort = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int gidIndex = args.length > 4 ? Integer.parseInt(args[4]) : 3;
        int rows = args.length > 5 ? Integer.parseInt(args[5]) : 128;

        long handle = RustRdmaNative.open(host, port, device, ibPort, gidIndex, "post",
                "{\"function\":\"INCREMENT\",\"field_index\":2,\"fields\":[\"INT64\",\"INT64\",\"DECIMAL_BYTES\",\"TIMESTAMP_MILLIS\",\"BYTES\",\"INT64\"]}");
        if (handle == 0) {
            throw new IOException("POST JNI open returned a null handle");
        }
        try {
            int received = 0;
            while (received < rows) {
                byte[][] batch = RustRdmaNative.receive(handle);
                for (byte[] payload : batch) {
                    if (payload.length < 58) {
                        throw new IOException("received serialized bid row is too short: " + payload.length);
                    }
                    ByteBuffer buffer = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN);
                    int frameLength = buffer.getInt();
                    if (frameLength != payload.length - 4) {
                        throw new IOException("invalid frame length: " + frameLength);
                    }
                    int op = buffer.getInt();
                    long rowId = buffer.getLong();
                    int nullBitmap = Byte.toUnsignedInt(buffer.get());
                    long auction = buffer.getLong();
                    long bidder = buffer.getLong();
                    int priceLength = buffer.getInt();
                    if (priceLength <= 0 || priceLength > 16 || priceLength > buffer.remaining()) {
                        throw new IOException("invalid DECIMAL price length: " + priceLength);
                    }
                    byte[] priceBytes = new byte[priceLength];
                    buffer.get(priceBytes);
                    long dateTime = buffer.getLong();
                    int extraLength = buffer.getInt();
                    if (extraLength < 0 || extraLength > buffer.remaining() - 8) {
                        throw new IOException("invalid extra length: " + extraLength);
                    }
                    byte[] extra = new byte[extraLength];
                    buffer.get(extra);
                    long latencyTs = buffer.getLong();

                    long expectedUnscaled = (received == 0 ? 0 : 12_345L + received) + 1;
                    if (op != 0
                            || rowId != received
                            || nullBitmap != 0
                            || auction != 1_000L + received
                            || bidder != 2_000L + received
                            || !new BigInteger(priceBytes).equals(BigInteger.valueOf(expectedUnscaled))
                            || dateTime != 1_700_000_000_000L + received
                            || !new String(extra, StandardCharsets.UTF_8).equals("smoke-" + received)
                            || latencyTs != 9_000L + received) {
                        throw new IOException("bid row mismatch at row " + received);
                    }
                    received++;
                }
            }
            System.out.println("POST smoke test received and validated " + received + " rows");
        } finally {
            RustRdmaNative.close(handle);
        }
    }
}
