
import java.io.*;
import java.nio.file.*;
import java.sql.*;
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class TFIDFApplication {

    // JDBC Connection Information for SQLite
    private static final String DB_URL = "jdbc:sqlite:corpus_db.sqlite";

    // File path for the stop words
    private static final String STOP_WORDS_FILE = "stop_words.txt";

    // List of stop words (loaded from file)
    private static final List<String> STOP_WORDS = loadStopWords();

    private static List<String> loadStopWords() {
        try {
            return Files.readAllLines(Paths.get(STOP_WORDS_FILE))
                        .stream()
                        .map(String::trim)
                        .filter(line -> !line.isEmpty() && !line.startsWith("#")) // Ignore empty lines and comments
                        .collect(Collectors.toList());
        } catch (IOException e) {
            System.err.println("Failed to load stop words: " + e.getMessage());
            return Collections.emptyList(); // Return an empty list if file reading fails
        }
    }

    public static void main(String[] args) {
        try (Connection connection = DriverManager.getConnection(DB_URL)) {
            initializeDatabase(connection); // Create the table if necessary
            List<Document> documents = loadDocumentsFromDatabase(connection);

            // Step 1: Tokenize and create a bag of words
            Map<Integer, List<String>> tokenizedDocs = tokenizeDocuments(documents);

            // Step 2: Clean the bag of words
            Map<Integer, List<String>> cleanedDocs = cleanDocuments(tokenizedDocs);

            // Step 3: Vectorize the documents
            Map<String, Map<Integer, Integer>> termDocumentMatrix = vectorize(cleanedDocs);

            // Step 4: Calculate IDF for each term
            Map<String, Double> idfValues = calculateIDF(termDocumentMatrix, documents.size());

            // Step 5: Calculate TF-IDF values
            Map<Integer, Map<String, Double>> tfidfMatrix = calculateTFIDF(cleanedDocs, termDocumentMatrix, idfValues);

            // Output results
            tfidfMatrix.forEach((docId, tfidfValues) -> {
                System.out.println("Document " + docId + ":");
                tfidfValues.forEach((term, value) -> System.out.println("  " + term + ": " + value));
            });
        } catch (SQLException e) {
            System.err.println("Database error: " + e.getMessage());
        }
    }

    private static void initializeDatabase(Connection connection) throws SQLException {
        String createTableSQL = """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL
            );
        """;

        try (Statement statement = connection.createStatement()) {
            statement.execute(createTableSQL);
        }

        String checkIfEmptySQL = "SELECT COUNT(*) AS count FROM documents";
        try (Statement statement = connection.createStatement();
             ResultSet resultSet = statement.executeQuery(checkIfEmptySQL)) {
            if (resultSet.next() && resultSet.getInt("count") == 0) {
                System.out.println("No documents found in the database. Inserting default data.");
                insertDefaultDocuments(connection);
            }
        }
    }

    private static void insertDefaultDocuments(Connection connection) throws SQLException {
        String insertSQL = """
            INSERT INTO documents (path) VALUES
            ('./docs/doc1.txt'),
            ('./docs/doc2.txt'),
            ('./docs/doc3.txt');
        """;

        try (Statement statement = connection.createStatement()) {
            statement.executeUpdate(insertSQL);
            System.out.println("Default documents have been inserted.");
        }
    }

    private static List<Document> loadDocumentsFromDatabase(Connection connection) throws SQLException {
        String query = "SELECT id, path FROM documents";
        try (Statement statement = connection.createStatement(); ResultSet resultSet = statement.executeQuery(query)) {
            List<Document> documents = new ArrayList<>();
            while (resultSet.next()) {
                int id = resultSet.getInt("id");
                String path = resultSet.getString("path");
                documents.add(new Document(id, path));
            }
            return documents;
        }
    }

    private static Map<Integer, List<String>> tokenizeDocuments(List<Document> documents) {
        Function<Document, List<String>> tokenizer = doc -> {
            try {
                return Arrays.asList(new String(Files.readAllBytes(new File(doc.getPath()).toPath())).split("\\W+"));
            } catch (IOException e) {
                System.err.println("Failed to read document: " + doc.getPath());
                return Collections.emptyList();
            }
        };

        return documents.stream().collect(Collectors.toMap(Document::getId, tokenizer));
    }

    private static Map<Integer, List<String>> cleanDocuments(Map<Integer, List<String>> tokenizedDocs) {
        Predicate<String> isNotStopWord = word -> !STOP_WORDS.contains(word);
        Predicate<String> isValidWord = word -> word.matches("[a-zàâçéèêëîïôûùüÿñæœ]+");

        return tokenizedDocs.entrySet().stream().collect(Collectors.toMap(
            Map.Entry::getKey,
            entry -> entry.getValue().stream()
                .map(String::toLowerCase)
                .filter(isNotStopWord.and(isValidWord))
                .collect(Collectors.toList())
        ));
    }

    private static Map<String, Map<Integer, Integer>> vectorize(Map<Integer, List<String>> cleanedDocs) {
        Map<String, Map<Integer, Integer>> termDocumentMatrix = new HashMap<>();

        cleanedDocs.forEach((docId, words) -> {
            Consumer<String> addWordToMatrix = word -> {
                termDocumentMatrix.putIfAbsent(word, new HashMap<>());
                termDocumentMatrix.get(word).put(docId, termDocumentMatrix.get(word).getOrDefault(docId, 0) + 1);
            };
            words.forEach(addWordToMatrix);
        });

        return termDocumentMatrix;
    }

    private static Map<String, Double> calculateIDF(Map<String, Map<Integer, Integer>> termDocumentMatrix, int totalDocs) {
        return termDocumentMatrix.entrySet().stream().collect(Collectors.toMap(
            Map.Entry::getKey,
            entry -> Math.log10((double) totalDocs / entry.getValue().size())
        ));
    }

    private static Map<Integer, Map<String, Double>> calculateTFIDF(Map<Integer, List<String>> cleanedDocs,
                                                                    Map<String, Map<Integer, Integer>> termDocumentMatrix,
                                                                    Map<String, Double> idfValues) {
        Map<Integer, Map<String, Double>> tfidfMatrix = new HashMap<>();

        cleanedDocs.forEach((docId, words) -> {
            Map<String, Double> tfidfValues = new HashMap<>();

            words.forEach(word -> {
                int termFrequency = termDocumentMatrix.get(word).getOrDefault(docId, 0);
                double tf = (double) termFrequency / words.size();
                double idf = idfValues.getOrDefault(word, 0.0);
                tfidfValues.put(word, tf * idf);
            });

            tfidfMatrix.put(docId, tfidfValues);
        });

        return tfidfMatrix;
    }

    static class Document {
        private final int id;
        private final String path;

        public Document(int id, String path) {
            this.id = id;
            this.path = path;
        }

        public int getId() {
            return id;
        }

        public String getPath() {
            return path;
        }
    }
}