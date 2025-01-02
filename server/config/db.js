const mongoose = require('mongoose');

const connectMongoDB = async () => {
    try {
        const connection = await mongoose.connect(process.env.MONGODB_URI);
        console.log(`MongoDB Connected: ${connection.connection.host}`);

        mongoose.connection.on('error', err => {
            console.error(`MongoDB connection error: ${err}`);
        });

        mongoose.connection.on('disconnected', () => {
            console.log('MongoDB disconnected');
        });

        process.on('SIGINT', async () => {
            await mongoose.connection.close();
            console.log('MongoDB connection closed through app termination');
            process.exit(0);
        });

    } catch (err) {
        console.log(`Error connecting to MongoDB: ${err.message}`);
        process.exit(1);
    }
};

module.exports = connectMongoDB;