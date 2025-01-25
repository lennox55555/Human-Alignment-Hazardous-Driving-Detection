import { useAuth } from "./useAuth";
import { SignInFormData } from "../utils/types";
import axios from "axios";

const useSignIn = () => {
    const { dispatch } = useAuth();

    const signIn = async (signInFormData: SignInFormData) => {
        try {
            const response = await axios.post('https://human-alignment-hazardous-driving.onrender.com/auth/signIn', signInFormData);
            console.log('User has successfully signed in!', response);

            localStorage.setItem('user', JSON.stringify(response.data));
            dispatch({ type: 'LOGIN', payload: response.data });
        } catch (error) {
            throw new Error(`An error has occurred while signing in: ${error}`);
        }
    };

    return { signIn }
}

export default useSignIn;