import styles from './SignInPage.module.css';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form'
import Row from 'react-bootstrap/Row';
import { SignInFormData } from '../../utils/types';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import useSignIn from '../../hooks/useSignIn';
import VideoPlayer from '../../components/video/VideoPlayer';

const SignInPage: React.FC = () => {
    const navigate = useNavigate();
    const { signIn } = useSignIn();
    const [formData, setFormData] = useState<SignInFormData>({
        email: '',
        password: '',
    });

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        console.log(formData);
        try {
            await signIn(formData)
        } catch (err: unknown) {
            console.log('An error has occured during registration', err)
        }
    };

    const handleNavigate = () => {
        navigate('/registration')
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setFormData((prevState) => ({
            ...prevState,
            [name]: value,
        }));
    };

    return (
        <div className={styles.container}>
            <div className={styles.containerWrapper}>
                <div className={styles.title}>
                    Welcome to the Human-Aligned Hazard Detection Survey.
                </div>
                <div className={styles.content}>
                    Human Aligned Hazardous Detection (HAHD) is a research initiative aimed at making driving behavior in autonomous systems more aligned with human decision-making.
                </div>

                {/* Video Player */}
                <div style={{ 
                    maxWidth: '800px', 
                    margin: '20px auto', 
                    padding: '10px',
                    width: '100%'
                }}>
                    <VideoPlayer />
                </div>

                <div>
                    <Form onSubmit={handleSubmit}>
                        <Form.Group className='mb-4' controlId="formGridEmail" >
                            <Form.Label>Email</Form.Label>
                            <Form.Control
                                type="email"
                                placeholder="name@example.com"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                required
                            />
                        </Form.Group>

                        <Row className='mb-4'>
                            <Form.Group as={Col} controlId="formGridPassword">
                                <Form.Label>Password</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Enter Password"
                                    name="password"
                                    value={formData.password}
                                    onChange={handleChange}
                                    required
                                />
                            </Form.Group>
                        </Row>

                        <Button className='mb-4' variant="dark" type="submit" style={{ width: '100%' }}>
                            Sign In
                        </Button>
                        <Button
                            variant="primary"
                            type="submit"
                            style={{ width: '100%' }}
                            onClick={handleNavigate}
                        >
                            Register
                        </Button>
                    </Form>
                </div>
            </div>
        </div>
    )
}

export default SignInPage;