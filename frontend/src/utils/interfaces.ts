export interface User {
    id: string;
    email: string;
    token: string;
}

export interface AuthState {
    user: any | null;
}

export type AuthAction = 
    | { type: 'LOGIN'; payload: any }
    | { type: 'LOGOUT' };

export interface AuthContextType {
    user: any | null;
    dispatch: React.Dispatch<AuthAction>;
}
