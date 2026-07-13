import { Client } from "pg";

const SQL = `
  CREATE EXTENSION IF NOT EXISTS pgcrypto;
  CREATE TABLE IF NOT EXISTS users(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR (255),
    password VARCHAR (255));
  CREATE TABLE dataset(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    filename VARCHAR(255),
    userId UUID REFERENCES users(id));
  CREATE TABLE project(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    filename VARCHAR(255),
    create_at TIMESTAMPTZ DEFAULT NOW(),
    userId UUID REFERENCES users(id));
  CREATE TABLE prebuiltdataset(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    image VARCHAR(255),
    description VARCHAR(255));
  INSERT INTO prebuiltdataset(name, image, description) VALUES
    ('Iris Dataset', 'Deceased', 'Containing 150 samples of iris flowers with four features each, used to classify them into three species: setosa, versicolor, and virginica. It’s small, clean, and ideal for beginners learning multiclass classification.'),
    ('Wine Dataset', 'Wine_Bar', 'Having 178 samples of wines with 13 chemical features, classified into three cultivars. It’s commonly used to practice feature analysis and multiclass classification models.'),
    ('Breast Cancer Dataset', 'Oncology', 'Including 569 samples of cell nuclei features, labeled as malignant or benign tumors. It’s a classic binary classification dataset used in medical machine learning applications.')
  `;
async function main() {
  console.log("seeding...");
  const arg = process.argv[2];
  const client = new Client({ connectionString: arg });
  await client.connect();
  await client.query(SQL);
  await client.end();
  console.log("Done!");
}
main();
