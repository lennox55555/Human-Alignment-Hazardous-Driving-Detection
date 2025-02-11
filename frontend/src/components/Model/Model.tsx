import { ModelProps } from "../../utils/interfaces";
import styles from './Model.module.css';

const Model: React.FC<ModelProps> = ({ title, description, results, limitations}) => {
    return (
        <div className={styles.container}>
            <div className={styles.wrapper}>
            <h1 className={styles.title}>{title}</h1>
            <p className={styles.description}>{description}</p>
            <div className={styles.section}>
                    <h4 className={styles.sectionTitle}>Results</h4>
                    <p>{results}</p>
                    <div className="container">
                      
                  </div>
                    
                  </div>
            <div className={styles.limitations}>
                <p>Limitations</p>
                <ul>
                    {limitations.map((limitation) =>
                        <li className={styles.limitationItem}>{limitation}</li>
                    )}
                </ul>
            </div>
            </div>
        </div>
    );
}

export default Model;