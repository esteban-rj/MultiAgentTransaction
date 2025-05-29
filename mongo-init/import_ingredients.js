print("Starting MongoDB initialization script...");

// Connect to the specified database (or 'test' if not specified)
const dbName = process.env.MONGO_INITDB_DATABASE || 'test';
const db = connect(`mongodb://localhost:27017/${dbName}`);

print(`Connected to database: ${dbName}`);

// Drop the collection if it already exists to ensure fresh import
if (db.getCollectionNames().includes('ingredientes_disponibles')) {
  db.ingredientes_disponibles.drop();
  print("Dropped existing 'ingredientes_disponibles' collection.");
}

// Read the CSV file content
// The CSV file is mounted at /data/available_ingredients.csv in the docker-compose.yml
const fs = require('fs'); // Import the Node.js filesystem module
const csvFilePath = "/data/available_ingredients.csv";
let fileContent;
try {
  fileContent = fs.readFileSync(csvFilePath, 'utf8'); // Use fs.readFileSync
  print(`Successfully read CSV file: ${csvFilePath}`);
} catch (e) {
  print(`Error reading CSV file ${csvFilePath}: ${e}`);
  quit(1); // Exit if file cannot be read
}

const lines = fileContent.split('\n');

if (lines.length <= 1) {
  print("CSV file is empty or only contains a header.");
  quit(1);
}

const headers = lines[0].split(',').map(header => header.trim());
const documents = [];

for (let i = 1; i < lines.length; i++) {
  const line = lines[i].trim();
  if (line === "") continue; // Skip empty lines

  const values = line.split(',');
  const doc = {};
  for (let j = 0; j < headers.length; j++) {
    let value = values[j] ? values[j].trim() : '';
    if (headers[j] === 'cantidad_gramos') {
      const parsedQuantity = parseInt(value, 10);
      if (isNaN(parsedQuantity)) {
        print(`Warning: Could not parse quantity '${value}' for ingredient '${values[1]}' on line ${i + 1}. Setting to 0.`);
        doc[headers[j]] = 0;
      } else {
        doc[headers[j]] = parsedQuantity;
      }
    } else if (headers[j] === 'uuid') {
      // Assuming uuid is the first column and might be numeric from the CSV
      // Store as string or number based on your preference, here as string
      doc[headers[j]] = String(value);
    } else {
      doc[headers[j]] = value;
    }
  }
  // Ensure 'nombre' and 'cantidad_gramos' fields exist as expected by the app
  if (!doc.hasOwnProperty('nombre') || !doc.hasOwnProperty('cantidad_gramos')) {
    print(`Warning: Line ${i + 1} is missing 'nombre' or 'cantidad_gramos'. Skipping.`);
    continue;
  }
  documents.push(doc);
}

if (documents.length > 0) {
  try {
    db.ingredientes_disponibles.insertMany(documents);
    print(`Successfully inserted ${documents.length} documents into 'ingredientes_disponibles' collection.`);
  } catch (e) {
    print(`Error inserting documents: ${e}`);
    quit(1);
  }
} else {
  print("No valid documents found to insert.");
}

print("MongoDB initialization script finished."); 