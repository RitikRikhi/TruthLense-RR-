import axios from 'axios';
const r = await axios.post('http://localhost:5000/api/analyze', {input: 'bleach cures covid'}).catch(e => e.response);
console.log('STATUS:', r?.status);
console.log('DATA:', JSON.stringify(r?.data, null, 2));
