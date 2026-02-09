package org.example.flinke2c.runtime;

@FunctionalInterface
interface OutputAccessor {
    Object get(Object result) throws Exception;
}
