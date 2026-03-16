import express from 'express';
import cors from 'cors';
import dotenv from "dotenv";
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

import connectDB from './config/db.js';
import analyzeRouter from './routes/analyzeRouter.js';

const app = express();

app.use(cors());
app.use(express.json());

// connectDB(); // Optional: enable when MongoDB is configured

app.use('/api', analyzeRouter);

// Serve static frontend from ../frontend
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendPath = path.join(__dirname, '..', 'frontend');

app.use(express.static(frontendPath));

app.get("/", (req, res) => {
  res.sendFile(path.join(frontendPath, 'index.html'));
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

