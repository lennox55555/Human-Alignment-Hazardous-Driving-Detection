import { useAuth } from "./useAuth";
import { RegistrationFormData } from "../utils/types";
import axios from "axios";

const useRegister = () => {
    const { dispatch } = useAuth();

    const registerUser = async (registrationFormData: RegistrationFormData) => {
        axios.post('http://localhost:3001/auth/register', registrationFormData)
            .then((payload) => {
                console.log('User has successfully registered!', payload);

                localStorage.setItem('user', JSON.stringify(payload.data));
                dispatch({ type: 'LOGIN', payload: payload.data });
            })
            .catch((err) => {
                console.log(`An error has occurred while registering the user, ${err}`);
            })
    }
    return { registerUser }
}

export default useRegister;

