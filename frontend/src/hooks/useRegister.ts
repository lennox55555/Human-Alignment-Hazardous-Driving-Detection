import { useAuth } from "./useAuth";
import { RegistrationFormData } from "../utils/types";
import axios from "axios";

const useRegister = () => {
    const { dispatch } = useAuth();

    const registerUser = async (registrationFormData: RegistrationFormData) => {
        try {
            const response = await axios.post('https://human-alignment-hazardous-driving.onrender.com/auth/register', registrationFormData);
            console.log('User has successfully registered!', response);

            localStorage.setItem('user', JSON.stringify(response.data));
            dispatch({ type: 'LOGIN', payload: response.data });

            return { success: true, data: response.data };
        } catch (err: any) {
            if (err.response) {
                console.log(`An error has occurred while registering the user: ${err.response.data.message}`);
                return { success: false, error: err.response.data.message };
            } else {
                console.log('An unexpected error occurred', err);
                return { success: false, error: 'An unexpected error occurred' };
            }
        }
    }
    
    return { registerUser }
}

export default useRegister;

