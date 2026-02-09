package org.example.flinke2c.runtime;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

final class PojoAccessors {
    private PojoAccessors() {}

    static OutputAccessor[] buildPojoAccessors(
            Class<?> returnType,
            String[] outputNames,
            Class<?>[] outFieldTypes) throws Exception {
        Map<String, Field> fields = new HashMap<>();
        for (Field field : returnType.getDeclaredFields()) {
            field.setAccessible(true);
            fields.put(normalizeName(field.getName()), field);
        }

        Map<String, Method> getters = new HashMap<>();
        for (Method method : returnType.getMethods()) {
            if (!isGetter(method)) {
                continue;
            }
            String prop = getterPropertyName(method);
            if (!prop.isEmpty()) {
                getters.put(normalizeName(prop), method);
            }
        }

        OutputAccessor[] accessors = new OutputAccessor[outputNames.length];
        for (int i = 0; i < outputNames.length; i++) {
            String name = outputNames[i];
            String key = normalizeName(name);
            Field field = fields.get(key);
            if (field != null) {
                accessors[i] = result -> field.get(result);
                outFieldTypes[i] = field.getType();
                continue;
            }
            Method getter = getters.get(key);
            if (getter != null) {
                accessors[i] = result -> getter.invoke(result);
                outFieldTypes[i] = getter.getReturnType();
                continue;
            }
            throw new IllegalArgumentException(
                    "No field or getter found for output name " + name
                            + " on " + returnType.getName());
        }

        return accessors;
    }

    private static String normalizeName(String name) {
        if (name == null) {
            return "";
        }
        return name.replace("_", "").toLowerCase(Locale.ROOT);
    }

    private static boolean isGetter(Method method) {
        if (method.getParameterCount() != 0) {
            return false;
        }
        if (method.getReturnType() == void.class) {
            return false;
        }
        String name = method.getName();
        if (name.startsWith("get") && name.length() > 3) {
            return true;
        }
        if (name.startsWith("is") && name.length() > 2
                && (method.getReturnType() == boolean.class
                || method.getReturnType() == Boolean.class)) {
            return true;
        }
        return false;
    }

    private static String getterPropertyName(Method method) {
        String name = method.getName();
        if (name.startsWith("get") && name.length() > 3) {
            return decapitalize(name.substring(3));
        }
        if (name.startsWith("is") && name.length() > 2) {
            return decapitalize(name.substring(2));
        }
        return "";
    }

    private static String decapitalize(String value) {
        if (value.isEmpty()) {
            return value;
        }
        char first = value.charAt(0);
        char lower = Character.toLowerCase(first);
        if (first == lower) {
            return value;
        }
        return lower + value.substring(1);
    }
}
