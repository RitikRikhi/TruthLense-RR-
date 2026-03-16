import axios from 'axios';
const searchWeb= async(claim)=>{
    console.log("Key:",process.env.NEWS_API_KEY);
try{
    const response=await axios.get(`https://newsapi.org/v2/everything?q=${claim}&pageSize=5&apiKey=${process.env.NEWS_API_KEY}`);
    console.log(response.data);
    const articles=response.data.articles;
    const sources=articles.map(article=>({
        title:article.title,
        url:article.url,
     source:article.source.name
    }));
    return sources;
}
catch(error){
    console.error("Error searching web:",error.response?.data||error.message);
    return [];
}
}
export default searchWeb;