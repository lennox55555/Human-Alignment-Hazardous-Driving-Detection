import { useAuth } from "./useAuth";
import { SignInFormData } from "../utils/types";
import axios from "axios";

const useSignIn = () => {
    const { dispatch } = useAuth();

    const signIn = async (signInFormData: SignInFormData) => {
        axios.post('http://localhost:3001/auth/signIn', (signInFormData))
            .then((payload) => {
                console.log('User has successfully signed in!', payload)

                localStorage.setItem('user', JSON.stringify(payload.data))
                dispatch({type: 'LOGIN', payload: payload.data});
            })
            .catch((err) => {
                console.log(`An error has occurred while signing in, ${err}`)
            })
    }
    return { signIn }
}

export default useSignIn;