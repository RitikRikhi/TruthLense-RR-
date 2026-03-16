import searchWeb from "./services/searchServices.js";

(async () => {
  const results = await searchWeb("delhi riots");
  console.log(results);
})();