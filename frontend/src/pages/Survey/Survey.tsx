import { Button } from "react-bootstrap";
import useSignOut from "../../hooks/useSignOut";
import { useAuth } from "../../hooks/useAuth";

const Survey: React.FC = () => {
    const { signOut } = useSignOut();
    const { user } = useAuth();

    return (
        <div>
            <h1>
                Survey
            </h1>
            {user && (
                <h4>
                    Welcome, {user.email} to the HAHD Survey!
                </h4>
            )}
            <Button variant="primary" onClick={signOut} style={{ padding: '6px 45px 6px 45px', marginTop: '25px' }}>
                Log Out
            </Button>
        </div>
    )
}

export default Survey;