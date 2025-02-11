import { useState } from 'react';
import { Card, Tabs, Tab, Button, Container } from 'react-bootstrap';
import styles from './Models.module.css'
import ModelVideoPlayer from '../../components/video/ModelVideoPlayer';

const ModelTabs = () => {
  const [activeKey, setActiveKey] = useState(0);
  const [videoPlaying, setVideoPlaying] = useState<boolean>(true);
  const [videoId, setVideoId] = useState('');
  const [startTime, setStartTime] = useState<number>(0);
  const [endTime, setEndTime] = useState<number>(0);

  const handleVideoComplete = () => {
    setVideoPlaying(false);
  }

  const handleVideoId = (videoId: string) => {
    setVideoId(videoId);
  }

  const handleFootageMetaData = (spacebarTimestamps: number[], startTime: number, endTime: number) => {
      setStartTime(startTime);
      setEndTime(endTime);
  }

  const models = [
    {
      title: 'Naive Model',
      description: 'A Naive Model',
      results: 'results here',
      limitations: ['first limitation', 'second limitation']
    },
    {
      title: 'Traditional Model',
      description: 'A Traditional Model',
      results: 'results here',
      limitations: ['first limitation', 'second limitation']
    },
    {
      title: 'Deep Learning Model',
      description: 'A Deep Learning Model',
      results: 'results here',
      limitations: ['first limitation', 'second limitation']
    }
  ];

  const handleNext = () => {
    const nextKey = (activeKey + 1) % models.length;
    setActiveKey(nextKey);
  };

  return (
    <Container fluid className={styles.container}>
      <Card className={styles.card}>
        <div className={styles.tabsContainer}>
          <Tabs
            activeKey={activeKey}
            onSelect={(k) => setActiveKey(Number(k))}
            className="border-bottom-0"
          >
            {models.map((model, index) => (
              <Tab
                key={index}
                eventKey={index}
                title={model.title}
              >
                <div className={styles.content}>
                  <h3 className={styles.title}>{model.title}</h3>
                  <p className={styles.description}>{model.description}</p>
                  
                  <div className={styles.section}>
                    <h4 className={styles.sectionTitle}>Results</h4>
                    <p>{model.results}</p>
                    <div className="container">
                      <div className="row">
                        <div className="col-md-6">
                          {/* Original Video  */}
                          <ModelVideoPlayer
                          onVideoComplete={handleVideoComplete}
                          passVideoId={handleVideoId}
                          passFootageMetaData={handleFootageMetaData}
                          />
                          <p className="text-center">Original Video</p>
                        </div>
                        <div className="col-md-6">
                          {/* Prediction */}
                          <ModelVideoPlayer
                          onVideoComplete={handleVideoComplete}
                          passVideoId={handleVideoId}
                          passFootageMetaData={handleFootageMetaData}
                          />
                          <p className="text-center">Output Video</p>
                        </div>
                      </div>
                  </div>
                    
                  </div>

                  <div className={styles.section}>
                    <h4 className={styles.sectionTitle}>Limitations</h4>
                    <ul className={styles.limitationsList}>
                      {model.limitations.map((limitation, index) => (
                        <li key={index} className={styles.limitationItem}>
                          {limitation}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className={styles.buttonContainer}>
                    <Button onClick={handleNext} variant='dark'>
                      Next Model
                    </Button>
                  </div>
                </div>
              </Tab>
            ))}
          </Tabs>
        </div>
      </Card>
    </Container>
  );
};

export default ModelTabs;