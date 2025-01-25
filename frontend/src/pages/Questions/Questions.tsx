import React, { useState, useEffect } from "react";
import { QuestionsFormData, QuestionsProps } from "../../utils/interfaces";
import { Button, Form } from "react-bootstrap";
import styles from './Questions.module.css'
import { useWebGazer } from "../../hooks/useWebGazer";
import usePostResults from "../../hooks/usePostResults";
import Profile from "../../components/Profile/Profile";
import UserStats from "../../components/UserStats/UserStats";
import Leaderboard from "../../components/Leaderboard/Leaderboard";
import axios from "axios";

const Questions: React.FC<QuestionsProps> = ({ onFormSumbitted, videoId, spacebarTimestamps, startTime, endTime }) => {
    const { finalGazeData, resetFinalGazeData } = useWebGazer();
    const { postResults } = usePostResults();
    const [userData, setUserData] = useState({
        email: '',
        surveysCompleted: 0,
        numRaffleEntries: 0
    });

    const [topUsers, setTopUsers] = useState<Array<{ email: string, numRaffleEntries: number }>>([]);
    const [currentUserRank, setCurrentUserRank] = useState<number>(0);
    const [formSubmitted, setFormSubmitted] = useState(false);

    useEffect(() => {
        const fetchTopUsers = async () => {
            try {
                const userItem = localStorage.getItem('user');
                if (userItem) {
                    const user = JSON.parse(userItem);

                    const response = await axios.get('https://human-alignment-hazardous-driving.onrender.com/survey/top-raffle-entries', {
                        params: {
                            currentUserEmail: user.email
                        }
                    });
                    setTopUsers(response.data.topUsers);
                    setCurrentUserRank(response.data.currentUserRank);
                }
            } catch (error) {
                console.error('Failed to fetch top users', error);
            }
        };

        const userItem = localStorage.getItem('user');

        if (userItem) {
            const user = JSON.parse(userItem);
            setUserData({
                email: user.email,
                surveysCompleted: user.surveysCompleted,
                numRaffleEntries: user.numRaffleEntries
            });
            fetchTopUsers();
        }
    }, []);


    const [formData, setFormData] = useState<QuestionsFormData>({
        hazardDetected: '',
        noDetectionReason: '',
        detectionConfidence: 0,
        hazardSeverity: 0,
        attentionFactors: [] as string[],
        spacebarTimestamps: spacebarTimestamps || [],
        startTime: startTime || 0,
        endTime: endTime || 0
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (onFormSumbitted) {
            onFormSumbitted();
            // Call-Back for Leaderboard Entry Animation
            setFormSubmitted(true);
        }

        const userItem = localStorage.getItem('user')

        if (userItem) {
            const user = JSON.parse(userItem)
            const userId = user.email

            // Update Survey Completion Count in Local Storage
            const updatedUser = { ...user, surveysCompleted: user.surveysCompleted + 1, numRaffleEntries: user.numRaffleEntries + 1 }
            localStorage.setItem('user', JSON.stringify(updatedUser))

            // Update local state
            setUserData({
                email: updatedUser.email,
                surveysCompleted: updatedUser.surveysCompleted,
                numRaffleEntries: updatedUser.numRaffleEntries
            });

            const windowDimensions = {
                width: window.innerWidth,
                height: window.innerHeight
            };

            const results = {
                userId: userId,
                videoId: videoId,
                windowDimensions: windowDimensions,
                gaze: finalGazeData,
                formData: formData,
                numSurveysCompleted: updatedUser.surveysCompleted
            }

            try {
                await postResults(results)
                resetFinalGazeData();
            } catch (error: unknown) {
                console.log('An error has occured while posting the survey results.')
            }
        } else {
            throw new Error('Authorization Error')
        }
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;

        if (type === 'checkbox') {
            const checked = (e.target as HTMLInputElement).checked;

            setFormData(prevState => {
                const currentFactors = prevState.attentionFactors || [];

                if (checked) {
                    return {
                        ...prevState,
                        [name]: [...currentFactors, value]
                    };
                } else {
                    return {
                        ...prevState,
                        [name]: currentFactors.filter(factor => factor !== value)
                    };
                }
            });
        } else {
            setFormData(prevState => ({
                ...prevState,
                [name]: value,
            }));
        }
    };

    return (
        <div className={styles.container}>
            <Leaderboard
                topUsers={topUsers}
                currentUser={{
                    email: userData.email,
                    numRaffleEntries: userData.numRaffleEntries
                }}
                currentUserRank={currentUserRank}
                formSubmitted={formSubmitted}
            />
            <UserStats
                surveysCompleted={userData.surveysCompleted}
                numRaffleEntries={userData.numRaffleEntries}
            />
            <Profile />
            <div className={styles.containerWrapper}>
                <div className={styles.title}>
                    Post-Simulation Survey
                </div>
                <div className={styles.content}>
                    <p className="mb-2 fst-italic"><span style={{ color: 'red' }}>*</span> Please complete the following assessment regarding the driving scenario you just observed. Upon submission, you will be presented with a new driving scenario to evaluate.</p>
                </div>
                <div>
                    <Form onSubmit={handleSubmit}>
                        <Form.Group className='mb-4'>
                            <Form.Label>Did you press the <b>spacebar</b> when you detected a hazardous instance?</Form.Label>
                            <div>
                                <Form.Check
                                    type="radio"
                                    label="Yes"
                                    name="hazardDetected"
                                    value="yes"
                                    checked={formData.hazardDetected === 'yes'}
                                    onChange={handleChange}
                                    required
                                />
                                <Form.Check
                                    type="radio"
                                    label="No"
                                    name="hazardDetected"
                                    value="no"
                                    checked={formData.hazardDetected === 'no'}
                                    onChange={handleChange}
                                />
                            </div>
                        </Form.Group>

                        {formData.hazardDetected === 'no' && (
                            <>
                                <Form.Group className='mb-4'>
                                    <Form.Label>If you did <b>NOT</b> press the spacebar during the video, why do you think <b>no hazards</b> were present?</Form.Label>
                                    <Form.Select
                                        name="noDetectionReason"
                                        value={formData.noDetectionReason}
                                        onChange={handleChange}
                                        required
                                    >
                                        <option value="">Select a reason</option>
                                        <option value="noHazards">There were no hazards</option>
                                        <option value="uncertain">I was unsure</option>
                                        <option value="subtleHazards">The hazards were too subtle</option>
                                    </Form.Select>
                                </Form.Group>

                                <Form.Group className='mb-4'>
                                    <Form.Label>On a scale from 1 (not at all) to 5 (very), how <b>confident</b> are you that the situation was <b>not hazardous?</b></Form.Label>
                                    <Form.Select
                                        name="detectionConfidence"
                                        value={formData.detectionConfidence}
                                        onChange={handleChange}
                                        required
                                    >
                                        <option value="">Select confidence level</option>
                                        {[1, 2, 3, 4, 5].map(num => (
                                            <option key={num} value={num}>{num}</option>
                                        ))}
                                    </Form.Select>
                                </Form.Group>
                            </>
                        )}

                        {formData.hazardDetected === 'yes' && (
                            <>
                                <Form.Group className='mb-4'>
                                    <Form.Label>On a scale of 1 to 5, how <b>dangerous</b> was the situation? (1 = No harm possible, 5 = Life-threatening or fatal)</Form.Label>
                                    <Form.Select
                                        name="hazardSeverity"
                                        value={formData.hazardSeverity}
                                        onChange={handleChange}
                                        required
                                    >
                                        <option value="">Select danger level</option>
                                        {[1, 2, 3, 4, 5].map(num => (
                                            <option key={num} value={num}>{num}</option>
                                        ))}
                                    </Form.Select>
                                </Form.Group>

                                <Form.Group className='mb-4'>
                                    <Form.Label>On a scale from 1 (not at all) to 5 (very), how <b>confident</b> are you that the hazard(s) you identified were <b>genuinely dangerous</b>?</Form.Label>
                                    <Form.Select
                                        name="detectionConfidence"
                                        value={formData.detectionConfidence}
                                        onChange={handleChange}
                                        required
                                    >
                                        <option value="">Select confidence level</option>
                                        {[1, 2, 3, 4, 5].map(num => (
                                            <option key={num} value={num}>{num}</option>
                                        ))}
                                    </Form.Select>
                                </Form.Group>

                                <Form.Group className='mb-4'>
                                    <Form.Label>What specific factors drew your attention to the hazard(s) you detected?</Form.Label>
                                    {[
                                        { label: 'Movement', value: 'motion' },
                                        { label: 'Speed', value: 'velocity' },
                                        { label: 'Proximity to another vehicle', value: 'proximity' },
                                        { label: 'Pedestrian', value: 'pedestrian' },
                                        { label: 'Environmental Conditions', value: 'environment' },
                                        { label: 'Road Work (Construction)', value: 'construction' },
                                        { label: 'Other', value: 'other' }
                                    ].map((factor) => (
                                        <Form.Check
                                            key={factor.value}
                                            type="checkbox"
                                            label={factor.label}
                                            name="attentionFactors"
                                            value={factor.value}
                                            checked={formData.attentionFactors?.includes(factor.value)}
                                            onChange={handleChange}
                                        />
                                    ))}
                                </Form.Group>
                            </>
                        )}
                        <Button className={styles.submitButton} variant="dark" type="submit">
                            Submit
                        </Button>
                    </Form>
                </div>
            </div>
        </div>
    );
}

export default Questions;