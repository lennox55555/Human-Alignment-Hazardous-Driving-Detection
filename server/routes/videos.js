const express = require('express');
const router = express.Router();
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getRandomVideo, getVideoById } = require('../services/videoService');

// Import the s3Client from videoService
const { s3Client } = require('../services/videoService');

// Test route to verify S3 connection
router.get('/test-s3', async (req, res) => {
  try {
    const listCommand = new ListObjectsV2Command({
      Bucket: process.env.S3_BUCKET_NAME,
      MaxKeys: 1
    });
    
    const response = await s3Client.send(listCommand);
    res.json({
      success: true,
      bucketName: process.env.S3_BUCKET_NAME,
      hasContents: !!response.Contents,
      itemCount: response.Contents?.length || 0,
      region: process.env.AWS_REGION,
      hasAccessKey: !!process.env.AWS_ACCESS_KEY_ID,
      hasSecretKey: !!process.env.AWS_SECRET_ACCESS_KEY
    });
  } catch (error) {
    console.error('Error in test-s3 route:', {
      message: error.message,
      stack: error.stack,
      code: error.code
    });
    res.status(500).json({
      success: false,
      error: error.message,
      code: error.code
    });
  }
});

router.get('/random', async (req, res) => {
  try {
    const video = await getRandomVideo();
    res.json(video);
  } catch (error) {
    console.error('Error in /random route:', {
      message: error.message,
      stack: error.stack,
      code: error.code
    });
    res.status(500).json({ 
      error: 'Failed to get video',
      details: error.message
    });
  }
});
router.get('/238', async (req, res) => {
  try {
    const video = await getVideoById();
    res.json(video);
  } catch (error) {
    console.error('Error in /238 route:', {
      message: error.message,
      stack: error.stack,
      code: error.code
    });
    res.status(500).json({ 
      error: 'Failed to get video',
      details: error.message
    });
  }
});

module.exports = router;
