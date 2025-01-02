require('dotenv').config();
const express = require('express');
const connectMongoDB = require('./config/db');
const auth = require('./routes/auth');
const survey = require('./routes/survey');
const videoRoutes = require('./routes/videos');
const cors = require('cors');
const app = express();

// Connect to MongoDB
connectMongoDB();

// Apply Middleware
app.use(cors());
app.use(express.json());

// Create Routes
app.use('/auth', auth);
app.use('/survey', survey);
app.use('/api/videos', videoRoutes);

app.get('/', (req, res) => {
    res.json('Welcome to the Human-Aligned Hazard Detection Survey');
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`express server listening on port ${PORT}`);
});