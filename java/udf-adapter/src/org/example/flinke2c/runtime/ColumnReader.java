package org.example.flinke2c.runtime;

@FunctionalInterface
interface ColumnReader {
    Object get(int row) throws Exception;
}
