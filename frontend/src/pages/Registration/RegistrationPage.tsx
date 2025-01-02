import styles from './RegistrationPage.module.css';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form'
import Row from 'react-bootstrap/Row';
import { RegistrationFormData } from '../../utils/types';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import useRegister from '../../hooks/useRegister';

const RegistrationPage: React.FC = () => {
    const navigate = useNavigate();
    const { registerUser } = useRegister();

    const [formData, setFormData] = useState<RegistrationFormData>({
        email: '',
        password: '',
        state: '',
        city: '',
        licenseAge: '',
        age: 0,
        ethnicity: '',
        carMakeModel: '',
        gender: '',
        speedingTicket: false,
        dui: false,
        visuallyImpaired: false
    });

    const [confirmedPassword, setConfirmedPassword] = useState('');

    const states = [
        { code: 'AL', name: 'Alabama' },
        { code: 'AK', name: 'Alaska' },
        { code: 'AZ', name: 'Arizona' },
        { code: 'AR', name: 'Arkansas' },
        { code: 'CA', name: 'California' },
        { code: 'CO', name: 'Colorado' },
        { code: 'CT', name: 'Connecticut' },
        { code: 'DE', name: 'Delaware' },
        { code: 'FL', name: 'Florida' },
        { code: 'GA', name: 'Georgia' },
        { code: 'HI', name: 'Hawaii' },
        { code: 'ID', name: 'Idaho' },
        { code: 'IL', name: 'Illinois' },
        { code: 'IN', name: 'Indiana' },
        { code: 'IA', name: 'Iowa' },
        { code: 'KS', name: 'Kansas' },
        { code: 'KY', name: 'Kentucky' },
        { code: 'LA', name: 'Louisiana' },
        { code: 'ME', name: 'Maine' },
        { code: 'MD', name: 'Maryland' },
        { code: 'MA', name: 'Massachusetts' },
        { code: 'MI', name: 'Michigan' },
        { code: 'MN', name: 'Minnesota' },
        { code: 'MS', name: 'Mississippi' },
        { code: 'MO', name: 'Missouri' },
        { code: 'MT', name: 'Montana' },
        { code: 'NE', name: 'Nebraska' },
        { code: 'NV', name: 'Nevada' },
        { code: 'NH', name: 'New Hampshire' },
        { code: 'NJ', name: 'New Jersey' },
        { code: 'NM', name: 'New Mexico' },
        { code: 'NY', name: 'New York' },
        { code: 'NC', name: 'North Carolina' },
        { code: 'ND', name: 'North Dakota' },
        { code: 'OH', name: 'Ohio' },
        { code: 'OK', name: 'Oklahoma' },
        { code: 'OR', name: 'Oregon' },
        { code: 'PA', name: 'Pennsylvania' },
        { code: 'RI', name: 'Rhode Island' },
        { code: 'SC', name: 'South Carolina' },
        { code: 'SD', name: 'South Dakota' },
        { code: 'TN', name: 'Tennessee' },
        { code: 'TX', name: 'Texas' },
        { code: 'UT', name: 'Utah' },
        { code: 'VT', name: 'Vermont' },
        { code: 'VA', name: 'Virginia' },
        { code: 'WA', name: 'Washington' },
        { code: 'WV', name: 'West Virginia' },
        { code: 'WI', name: 'Wisconsin' },
        { code: 'WY', name: 'Wyoming' }
    ];

    const ethnicities = [
        "Asian",
        "Black or African American",
        "Hispanic or Latino",
        "Native American or Alaska Native",
        "Native Hawaiian or Other Pacific Islander",
        "White",
        "Middle Eastern or North African",
        "Multiracial",
        "Other",
        "Prefer not to say",
    ];

    const handleNavigate = () => {
        navigate('/signin')
    }

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        try {
            await registerUser(formData);
            navigate('/signin')
        } catch (err: unknown) {
            console.log('An error has occured during registration', err)
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;

        let updatedValue: any;

        if (type === 'checkbox') {
            updatedValue = (e.target as HTMLInputElement).checked;
        } else {
            updatedValue = value;
        }

        setFormData((prevState) => ({
            ...prevState,
            [name]: updatedValue,
        }));
    };

    const handleConfirmPassword = (e: React.ChangeEvent<HTMLInputElement>) => {
        const password = e.target.value;
        setConfirmedPassword(password)
    }

    return (
        <div className={styles.container}>
            <div className={styles.containerWrapper}>
                <div className={styles.title}>
                    Welcome to the Human-Aligned Hazard Detection Survey.
                </div>
                <div className={styles.content}>
                    Human Aligned Hazardous Detection (HAHD) is a research initiative aimed at making driving behavior in autonomous systems more aligned with human decision-making.
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

                            <Form.Group as={Col} controlId="formGridPassword">
                                <Form.Label>Confirm Password</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Confirm Password"
                                    name="confirmedPassword"
                                    value={confirmedPassword}
                                    onChange={handleConfirmPassword}
                                    required
                                    disabled={formData.password ? false : true}
                                    isInvalid={confirmedPassword && formData.password && confirmedPassword != formData.password ? true : false}
                                />
                                <Form.Control.Feedback type="invalid">
                                    Passwords do not match.
                                </Form.Control.Feedback>
                            </Form.Group>
                        </Row>

                        <Row className="mb-4">
                            <Form.Group as={Col} controlId="formGridCity">
                                <Form.Label>City</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter City"
                                    name="city"
                                    value={formData.city}
                                    onChange={handleChange}
                                />
                            </Form.Group>

                            <Form.Group as={Col} controlId="formGridState">
                                <Form.Label>State</Form.Label>
                                <Form.Control
                                    as="select"
                                    name="state"
                                    value={formData.state}
                                    onChange={handleChange}
                                >
                                    <option value="">Select...</option>
                                    {states.map((state, index) => (
                                        <option key={index} value={state.name}>
                                            {state.name}
                                        </option>
                                    ))}
                                </Form.Control>
                            </Form.Group>
                        </Row>

                        <Row className="mb-4">
                            {/* Current Age */}
                            <Form.Group as={Col} controlId="formAge">
                                <Form.Label>What is your age?</Form.Label>
                                <Form.Control
                                    type="number"
                                    placeholder="Enter age"
                                    name="age"
                                    min="0"
                                    max="150"
                                    value={formData.age}
                                    onChange={(e) => {
                                        const value = Math.max(0, Math.min(150, Number(e.target.value))); // Restrict to 0-150
                                        setFormData({ ...formData, age: value });
                                    }}
                                />
                            </Form.Group>

                            {/* Gender */}
                            <Form.Group as={Col} controlId="formGender">
                                <Form.Label>What is your gender?</Form.Label>
                                <Form.Select name="gender" value={formData.gender} onChange={handleChange}>
                                    <option value="">Select...</option>
                                    <option value="male">Male</option>
                                    <option value="female">Female</option>
                                    <option value="prefer-not-to-say">Prefer not to say</option>
                                </Form.Select>
                            </Form.Group>
                        </Row>

                        {/* Ethnicity */}
                        <Form.Group className="mb-4" controlId="formEthnicity">
                            <Form.Label>What is your ethnicity?</Form.Label>
                            <Form.Control
                                as="select"
                                name="ethnicity"
                                value={formData.ethnicity}
                                onChange={handleChange}
                            >
                                <option value="">Select...</option>
                                {ethnicities.map((ethnicity, index) => (
                                    <option key={index} value={ethnicity}>
                                        {ethnicity}
                                    </option>
                                ))}
                            </Form.Control>
                        </Form.Group>

                        {/* License Age */}
                        <Form.Group className="mb-4" controlId="formLicenseAge">
                            <Form.Label>How many years have you been driving for?</Form.Label>
                            <Form.Control
                                type="number"
                                placeholder="Enter age"
                                name="licenseAge"
                                value={formData.licenseAge}
                                onChange={handleChange}
                            />
                        </Form.Group>

                        {/* Make/Model of Car */}
                        <Form.Group className="mb-4" controlId="formCarMakeModel">
                            <Form.Label>What is the make/model of your car?</Form.Label>
                            <Form.Control
                                type="text"
                                placeholder="Enter make/model"
                                name="carMakeModel"
                                value={formData.carMakeModel}
                                onChange={handleChange}
                            />
                        </Form.Group>

                        {/* Visually Impaired */}
                        <Form.Group className="mb-4" controlId="formVisuallyImpaired">
                            <Form.Label>Are you visually impaired?</Form.Label>
                            <div>
                                <Form.Check
                                    type="radio"
                                    label="Yes"
                                    name="visuallyImpaired"
                                    value="true"
                                    checked={formData.visuallyImpaired === true}
                                    onChange={(e) => {
                                        const value = e.target.value === "true";
                                        setFormData(prevState => ({
                                            ...prevState,
                                            visuallyImpaired: value
                                        }));
                                    }}
                                />
                                <Form.Check
                                    type="radio"
                                    label="No"
                                    name="visuallyImpaired"
                                    value="false"
                                    checked={formData.visuallyImpaired === false}
                                    onChange={(e) => {
                                        const value = e.target.value === "true";
                                        setFormData(prevState => ({
                                            ...prevState,
                                            visuallyImpaired: value
                                        }));
                                    }}
                                />
                            </div>
                        </Form.Group>

                        {/* DUI */}
                        <Form.Group className="mb-4" controlId="formDUI">
                            <Form.Label>Have you ever had a DUI?</Form.Label>
                            <div>
                                <Form.Check
                                    type="radio"
                                    label="Yes"
                                    name="dui"
                                    value="true"
                                    checked={formData.dui === true}
                                    onChange={(e) => {
                                        const value = e.target.value === "true";
                                        setFormData(prev => ({
                                            ...prev,
                                            dui: value
                                        }));
                                    }}
                                />
                                <Form.Check
                                    type="radio"
                                    label="No"
                                    name="dui"
                                    value="false"
                                    checked={formData.dui === false}
                                    onChange={(e) => {
                                        const value = e.target.value === "true";
                                        setFormData(prevState => ({
                                            ...prevState,
                                            dui: value
                                        }));
                                    }}
                                />
                            </div>
                        </Form.Group>

                        {/* Speeding Ticket */}
                        <Form.Group className="mb-4" controlId="formSpeedingTicket">
                            <Form.Label>Have you ever had a speeding ticket?</Form.Label>
                            <div>
                                <Form.Check
                                    type="radio"
                                    label="Yes"
                                    name="speedingTicket"
                                    value="true"
                                    checked={formData.speedingTicket === true}
                                    onChange={(e) => {
                                        const value = e.target.value === "true";
                                        setFormData(prevState => ({
                                            ...prevState,
                                            speedingTicket: value
                                        }));
                                    }}
                                />
                                <Form.Check
                                    type="radio"
                                    label="No"
                                    name="speedingTicket"
                                    value="false"
                                    checked={formData.speedingTicket === false}
                                    onChange={(e) => {
                                        const value = e.target.value === "true";
                                        setFormData(prevState => ({
                                            ...prevState,
                                            speedingTicket: value
                                        }));
                                    }}
                                />
                            </div>
                        </Form.Group>

                        <div className={styles.buttonGroup}>
                            <Button variant="dark" type="submit" style={{ padding: '6px 45px 6px 45px', marginTop: '25px' }}>
                                Submit
                            </Button>
                            <Button variant="danger" onClick={handleNavigate} style={{ padding: '6px 45px 6px 45px', marginTop: '25px' }}>
                                Cancel
                            </Button>
                        </div>
                    </Form>
                </div>
            </div>
        </div>
    )
}

export default RegistrationPage;