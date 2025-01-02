import React, { useState, useEffect } from 'react';

interface VideoData {
    url: string;
    videoId: string;
}

const VideoPlayer: React.FC = () => {
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchRandomVideo = async (): Promise<void> => {
        try {
            setLoading(true);
            const response = await fetch('http://localhost:3001/api/videos/random');
            if (!response.ok) {
                throw new Error('Failed to fetch video');
            }
            const data: VideoData = await response.json();
            console.log('Received video data:', data);
            setVideoUrl(data.url);
            setError(null);
        } catch (err) {
            setError('Failed to load video');
            console.error('Error fetching video:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchRandomVideo();
    }, []);

    if (loading) return <div>Loading video...</div>;
    if (error) return <div>{error}</div>;

    return (
        <div className="video-container">
            {videoUrl && (
                <video 
                    controls 
                    width="100%" 
                    style={{ maxWidth: '800px' }}
                    key={videoUrl}
                >
                    <source src={videoUrl} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
            )}
            <button 
                onClick={fetchRandomVideo}
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
                style={{ marginTop: '10px' }}
            >
                Load Another Video
            </button>
        </div>
    );
};

export default VideoPlayer;