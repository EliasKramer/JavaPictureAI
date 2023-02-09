package ai.util;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;

public class Hasher {
    public static String getHash(String input) {
        MessageDigest digest = null;
        try {
            //"SHA-256", "SHA-512", "SHA-384", "SHA-224", "SHA-1", "MD5"
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        byte[] byteOfTextToHash = input.getBytes(StandardCharsets.UTF_8);
        byte[] hashedByteArray = digest.digest(byteOfTextToHash);

        return Base64.getEncoder().encodeToString(hashedByteArray);
    }
}
