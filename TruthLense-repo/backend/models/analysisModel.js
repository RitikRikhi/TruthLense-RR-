import mongoose from 'mongoose';
const analysisSchema = new mongoose.Schema({
    claim:{
        type:String,
        required:true
    },

    source:[
        {
            title:String,
            url:String,
            credibility:Number
        }
    ],

    verdict:{
        type:String,
    },

    createdAt:{
        type:Date,
        default:Date.now
    }
});

const Analysis=mongoose.model("Analysis",analysisSchema)

export default Analysis;