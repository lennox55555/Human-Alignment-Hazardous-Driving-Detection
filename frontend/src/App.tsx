import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import './App.css'
import SignInPage from './pages/SignIn/SignInPage';
import RegistrationPage from './pages/Registration/RegistrationPage';
import LandingPage from './pages/LandingPage/LandingPage';
import Calibration from './pages/Calibration/Calibration';
import Survey from './pages/Survey/Survey';
import Models from './pages/Models/Models';
import { useWebGazer } from './hooks/useWebGazer';

const App: React.FC = () => {

  const { user } = useAuth();
  const { isCalibrated } = useWebGazer();

  return (
    <Routes>
      <Route path='/' element={ user ? <Navigate to='/landingpage' /> : <SignInPage />} />
      <Route path='/models' element={<Models />} />
      <Route path='/registration' element={<RegistrationPage />} />
      <Route path='/landingpage' 
                element={ !user ? <Navigate to='/' /> : <LandingPage /> } 
            />
            <Route path='/calibration'
                element={
                    !user ? <Navigate to='/' /> :
                    isCalibrated ? <Navigate to='/survey' /> : <Calibration />
                } 
            />
            <Route path='/survey' 
                element={
                    !user ? <Navigate to='/' /> :
                    !isCalibrated ? <Navigate to='/landingpage' /> : <Survey />
                } 
            />
    </Routes>
  )
}

export default App
