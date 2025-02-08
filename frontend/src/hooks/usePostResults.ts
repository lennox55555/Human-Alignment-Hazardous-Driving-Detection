import { SurveyResults } from "../utils/interfaces"
import axios from "axios"

const usePostResults = () => {
    const postResults = async (results: SurveyResults) => {
        axios.post('https://human-alignment-hazardous-driving.onrender.com/survey/results', results)
            .then(() => {
                console.log('Survey results have been posted to MongoDB.');
            })
            .catch((err) => {
                console.log(`An error has occurred while posting survey results, ${err}`);
            })
    }

    return { postResults };
}

export default usePostResults;