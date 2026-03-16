import { GoogleGenerativeAI } from '@google/generative-ai';
const genAI = new GoogleGenerativeAI('AIzaSyB_vwxgJQEEPqde-K3UC0Q8eRjRGDiirlo');
const m = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
m.generateContent('Write exactly 4 short bullet points. Each bullet must start with either "-" or "*". Return ONLY the 4 lines.').then(r=>{
  const text = r.response.text();
  console.log("RAW TEXT:\n", text);
  const lines = text.split('\n').map(l=>l.trim()).filter(l=>l.startsWith('-')||l.startsWith('*')).slice(0, 4);
  console.log("FILTERED LINES:", lines);
}).catch(console.error);
