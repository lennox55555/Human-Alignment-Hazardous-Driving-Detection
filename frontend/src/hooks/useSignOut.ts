import { useAuth } from "./useAuth";

const useSignOut = () => {
    const { dispatch } = useAuth();

    const signOut = () => {
        dispatch({ type: 'LOGOUT'})
    }

    return { signOut }
}

export default useSignOut;