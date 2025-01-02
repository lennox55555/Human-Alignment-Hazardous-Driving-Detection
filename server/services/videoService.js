const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

// Log AWS Configuration
console.log('AWS Configuration:', {
    region: process.env.AWS_REGION,
    bucket: process.env.S3_BUCKET_NAME,
    hasAccessKey: !!process.env.AWS_ACCESS_KEY_ID,
    hasSecretKey: !!process.env.AWS_SECRET_ACCESS_KEY
});

const s3Client = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
});

async function getRandomVideo() {
    try {
        console.log('Attempting to fetch video from bucket:', process.env.S3_BUCKET_NAME);
        const listCommand = new ListObjectsV2Command({
            Bucket: process.env.S3_BUCKET_NAME,
        });
        
        const { Contents } = await s3Client.send(listCommand);
        console.log('Found videos in bucket:', Contents?.length);
        
        if (!Contents || Contents.length === 0) {
            throw new Error('No videos found in bucket');
        }

        const randomIndex = Math.floor(Math.random() * Contents.length);
        const randomVideo = Contents[randomIndex];
        
        const getObjectCommand = new GetObjectCommand({
            Bucket: process.env.S3_BUCKET_NAME,
            Key: randomVideo.Key
        });
        
        const presignedUrl = await getSignedUrl(s3Client, getObjectCommand, {
            expiresIn: 3600
        });
        
        console.log('Generated presigned URL for video:', randomVideo.Key);
        
        return {
            url: presignedUrl,
            videoId: randomVideo.Key
        };
    } catch (error) {
        console.error('Detailed error in getRandomVideo:', error);
        throw error;
    }
}

module.exports = { getRandomVideo };