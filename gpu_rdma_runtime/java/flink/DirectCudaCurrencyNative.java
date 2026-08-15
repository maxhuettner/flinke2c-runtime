/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.runtime.functions.table.externalruntime;

import java.nio.ByteBuffer;

/**
 * JNI entry point for the direct CUDA currency conversion kernel, used by
 * {@link CudaCurrencyConversionOperator}.
 *
 * <p>This is a separate bridge from {@code org.example.flinke2c
 * .CurrencyConversionGpuNative} even though both ultimately call into the
 * same {@code direct_currency_conversion_jni.cu} logic (see that file's
 * "Bridge 1"/"Bridge 2" comments): this class is expected to be compiled
 * into the Flink distribution itself, the same way {@link RustRdmaNative} is
 * for the RDMA path, while {@code CurrencyConversionGpuNative} ships in the
 * separate {@code flinke2c} user JAR. Flink's own classloader generally
 * cannot see classes from a separately deployed user JAR, so this class
 * exists instead of reusing that one directly. Loading the same {@code .so}
 * from two different classloaders in one JVM process can also fail with
 * {@code UnsatisfiedLinkError: Native Library ... already loaded in another
 * classloader} — if a TaskManager might run both
 * {@code CurrencyConversionFunctionGpu} and {@link CudaCurrencyConversionOperator}
 * over its lifetime, verify this doesn't happen in your deployment before
 * relying on both simultaneously.
 */
final class DirectCudaCurrencyNative {
    static {
        System.loadLibrary("flinke2c_currency_conversion_gpu");
    }

    private DirectCudaCurrencyNative() {}

    static native long create(int cudaDevice, int batchCapacity, int pipelineDepth, int threadsPerBlock);

    static native ByteBuffer inputBuffer(long handle, int lane);

    static native ByteBuffer outputBuffer(long handle, int lane);

    static native void submitBatch(long handle, int lane, int count);

    static native void waitBatch(long handle, int lane);

    static native void destroy(long handle);
}
