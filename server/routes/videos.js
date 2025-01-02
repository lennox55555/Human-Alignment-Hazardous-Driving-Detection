const express = require('express');
const router = express.Router();
const { getRandomVideo } = require('../services/videoService');

router.get('/random', async (req, res) => {
    try {
        const video = await getRandomVideo();
        res.json(video);
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Failed to get video' });
    }
});

module.exports = router;