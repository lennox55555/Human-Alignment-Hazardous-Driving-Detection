import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import './App.css'
import SignInPage from './pages/SignIn/SignInPage';
import RegistrationPage from './pages/Registration/RegistrationPage';
import Survey from './pages/Survey/Survey';

const App: React.FC = () => {

  const { user } = useAuth();

  return (
    <Routes>
      <Route path='/signin' element={ user ? <Navigate to='/survey' /> : <SignInPage />} />
      <Route path='/registration' element={<RegistrationPage />} />
      <Route path='/survey' element={ user ? <Survey /> : <Navigate to='/signin' />} />
    </Routes>
  )
}

export default App
