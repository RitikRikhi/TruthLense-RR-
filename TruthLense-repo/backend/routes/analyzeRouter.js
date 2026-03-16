import express from 'express';

import analysisController from '../controller/analyzeController.js';
const router = express.Router();
router.post('/analyze',analysisController);
export default router;