package org.example.flinke2c;

import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.functions.ScalarFunction;

import java.math.BigDecimal;

/**
 * A scalar function that converts the bid price from USD to EUR.
 */
public class CurrencyConversionFunction extends ScalarFunction {
    private static final BigDecimal CONVERSION_FACTOR = new BigDecimal("0.908");

    public @DataTypeHint("DECIMAL(23,3)") BigDecimal eval(
            @DataTypeHint("DECIMAL(23,3)") BigDecimal price) {
        return price == null ? null : price.multiply(CONVERSION_FACTOR);
    }
}
