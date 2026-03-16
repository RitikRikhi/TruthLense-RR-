import mongoose from 'mongoose';
const connectDB=async()=>{
    try{
        await mongoose.connect(process.env.MONGO_URI);
        console.log("MongoDB connected successfully")
    }catch(error){
        console.warn("MongoDB connection failed — running without DB persistence:", error.message);
        // Do NOT exit — server can still handle requests without MongoDB
    }
}

export default connectDB;