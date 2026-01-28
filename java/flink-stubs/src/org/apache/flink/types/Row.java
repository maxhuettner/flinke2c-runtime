package org.apache.flink.types;

import java.util.Arrays;

/**
 * Minimal stub of Flink's Row type for UDF compilation and proxy-side reflection.
 */
public final class Row {
    private final Object[] fields;

    private Row(int arity) {
        this.fields = new Object[arity];
    }

    public static Row of(Object... values) {
        Row row = new Row(values.length);
        System.arraycopy(values, 0, row.fields, 0, values.length);
        return row;
    }

    public int getArity() {
        return fields.length;
    }

    public Object getField(int pos) {
        return fields[pos];
    }

    public void setField(int pos, Object value) {
        fields[pos] = value;
    }

    @Override
    public String toString() {
        return Arrays.toString(fields);
    }
}

