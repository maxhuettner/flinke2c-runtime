package org.example.flinke2c.runtime;

@FunctionalInterface
interface ValueParser {
    Object parse(String value);
}
