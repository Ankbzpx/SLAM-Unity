using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

public class GraphSLAM : MonoBehaviour
{
    [SerializeField]
    int optimizeLength = 100;

    int historyLength = 10000;

    Matrix<float> R = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(0.2f, 2), Mathf.Pow(0.2f, 2) });
    Matrix<float> Q = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(0.2f, 2), Mathf.Pow(0.2f, 2), Mathf.Pow(Mathf.Deg2Rad * 5f, 2) });

    // True noise free, actual with noise
    Vector<float> latest_X_true, latest_X_actual;

    Queue<Vector<float>> robotTrace_true = new Queue<Vector<float>>();
    Queue<Vector<float>> robotTrace_actual = new Queue<Vector<float>>();
    Queue<Dictionary<int, Vector<float>>> landmarkTrace = new Queue<Dictionary<int, Vector<float>>>();

    float targetDis, targetAng = 0f;

    //Car parameters
    [SerializeField]
    float velocity = 5f;

    float threshold = 0.01f;
    float[] measureRange = new float[] { 10f, 120, Mathf.PI / 4 };
    Normal normal = Normal.WithMeanPrecision(0, 1);

    // Dictionary that stores associated landmarks
    public Dictionary<int, Vector<float>> landmarkDictionary =
        new Dictionary<int, Vector<float>>()
        {
                {0, Vector<float>.Build.DenseOfArray(new float[] {5, 5})},
                {1, Vector<float>.Build.DenseOfArray(new float[] {20, 10})},
                {2, Vector<float>.Build.DenseOfArray(new float[] {15, 15})},
                {3, Vector<float>.Build.DenseOfArray(new float[] {0, 15})}
        };

    // Start is called before the first frame update
    void Start()
    {
        
    }

    void FixedUpdate()
    {
        if (targetDis >= threshold)
        {
            float deltaDis = velocity * Mathf.Lerp(0, targetDis, Time.fixedDeltaTime);

            deltaDis = deltaDis > 0.1f ? deltaDis : 0.1f;

            MoveWorldPoints(deltaDis, targetAng);

            targetDis -= deltaDis;
            targetAng = 0f;
        }
    }

    void MoveWorldPoints(float deltaDis, float deltaAng)
    {
        //Shall not happen
        if (robotTrace_true.Count != robotTrace_actual.Count)
            return;

        //Avoid Overflow
        if(robotTrace_true.Count >= historyLength)
        {
            //Dequeue
            robotTrace_true.Dequeue();
            robotTrace_actual.Dequeue();
        }

        //Start by motion update
        latest_X_true = MotionUpdateTrue(latest_X_true, deltaDis, deltaAng);
        latest_X_actual = MotionUpdateActual(latest_X_true, deltaDis, deltaAng);
        
        landmarkTrace.Enqueue(ObserveLandmarks(latest_X_true));
        robotTrace_true.Enqueue(latest_X_true.Clone());
        robotTrace_actual.Enqueue(latest_X_actual.Clone());
    }

    public void Optimize()
    {
        Dictionary<int, Vector<float>>[] landmarkTrace_array = landmarkTrace.ToArray();
        Vector<float> robotTrace_actual_vector = QueueToVector(robotTrace_actual);

        Vector<float>  b = Vector<float>.Build.SparseOfArray(new float[robotTrace_actual_vector.Count]);
        Matrix<float>  H = Matrix<float>.Build.SparseOfArray(new float [robotTrace_actual_vector.Count, robotTrace_actual_vector.Count]);

        //For each combination
        foreach (int[] indexPair in GetIndexPairs(robotTrace_actual_vector.Count/3))
        {
            Vector<float> X_i = robotTrace_actual_vector.SubVector(3* indexPair[0], 3);
            Dictionary<int, Vector<float>> lmObs_i = landmarkTrace_array[indexPair[0]];

            Vector<float> X_j = robotTrace_actual_vector.SubVector(3 * indexPair[1], 3);
            Dictionary<int, Vector<float>> lmObs_j = landmarkTrace_array[indexPair[1]];

            //Loop through all landmarks
            foreach (int lmIdx in landmarkDictionary.Keys)
            {
                //If same landmark is observed at different robot pos
                if (lmObs_i.ContainsKey(lmIdx) && lmObs_j.ContainsKey(lmIdx))
                {
                    Vector<float> err = GetError(X_i, X_j, lmObs_i[lmIdx], lmObs_j[lmIdx]);
                    Matrix<float> A = GetJacobianA(X_i, lmObs_i[lmIdx]);
                    Matrix<float> B = -GetJacobianA(X_j, lmObs_j[lmIdx]);

                    //Omega, update H, b
                }
            }
        }

        //Solve
    }

    Vector<float> QueueToVector(Queue<Vector<float>> robotTrace_actual)
    {
        Vector<float> robStateVector = Vector<float>.Build.DenseOfArray(new float[robotTrace_actual.Count * 3]);

        int count = 0;

        while (robotTrace_actual.Count > 0)
        {
            Vector<float> state = robotTrace_actual.Dequeue();
            robStateVector.SetSubVector(3*count, 3, state);
            count++;
        }

        return robStateVector;
    }

    Queue<Vector<float>> VectorToQueue(Vector<float> robStateVector)
    {
        Queue<Vector<float>> robotTrace_actual = new Queue<Vector<float>>();

        for (int i = 0; i < robStateVector.Count/3; i++)
        {
            robotTrace_actual.Enqueue(robStateVector.SubVector(3*i, 3));
        }

        return robotTrace_actual;
    }


    Vector<float> GetError(Vector<float> X_i, Vector<float> X_j, Vector<float> z_i, Vector<float> z_j)
    {
        float ang_j = ClampRad(X_j[2] + z_j[1]);
        float ang_i = ClampRad(X_i[2] + z_i[1]);


        float e_x = X_j[0] + z_j[0] * Mathf.Cos(ang_j) - (X_i[0] + z_i[0] * Mathf.Cos(ang_i));
        float e_y = X_j[1] + z_j[0] * Mathf.Sin(ang_j) - (X_i[1] + z_i[0] * Mathf.Sin(ang_i));
        float e_a = ang_j - ang_i;

        return Vector<float>.Build.DenseOfArray(new float[3] {e_x , e_y, e_a});
    }

    Matrix<float> GetJacobianA(Vector<float> X, Vector<float> z)
    {
        float ang = ClampRad(X[2] + z[1]);

        return Matrix<float>.Build.DenseOfArray(new float[3, 3] 
            {{-1, 0, z[0]*Mathf.Sin(ang)}, 
             {0, -1, -z[0]*Mathf.Cos(ang)}, 
             {0, 0, -1 }});
    }

    void InitializeRobotState(float x, float y, float theta)
    {
        latest_X_true = Vector<float>.Build.DenseOfArray(new float[] { x, y, theta });
        latest_X_actual = latest_X_true.Clone();

        landmarkTrace.Enqueue(ObserveLandmarks(latest_X_true));
        robotTrace_true.Enqueue(latest_X_true.Clone());
        robotTrace_actual.Enqueue(latest_X_actual.Clone());
    }

    public void InitializeLandmarks(int numOfLandmarks, float range_x, float range_y)
    {
        landmarkDictionary.Clear();
        for (int i = 0; i < numOfLandmarks; i++)
        {
            float x = Random.Range(0, range_x * 0.8f);
            float y = Random.Range(0, range_y * 0.6f);

            landmarkDictionary.Add(i, Vector<float>.Build.DenseOfArray(new float[] { x, y }));
            //visualizer.VisualizeTrueLandmarks(i, x, y);
        }
    }

    public void SetTargetWorldPoints(float world_x, float world_y)
    {
        CalculateTargetPose(latest_X_actual, world_x, world_y, out targetDis, out targetAng);
    }

    void CalculateTargetPose(Vector<float> X_true, float x_world, float y_world, out float t_Dis, out float t_Ang)
    {
        float x_r = X_true[0];
        float y_r = X_true[1];
        float theta_r = X_true[2];

        // Calculate distance and theta from target world point
        t_Dis = Mathf.Sqrt(Mathf.Pow(x_world - x_r, 2) + Mathf.Pow(y_world - y_r, 2));
        t_Ang = Mathf.Atan2(y_world - y_r, x_world - x_r) - theta_r;
        t_Ang = ClampRad(t_Ang);

    }

    // Only update robots
    Vector<float> MotionUpdateTrue(Vector<float> X_true, float dis, float ang)
    {
        float theta_r_est = ClampRad(ang + X_true[2]);
        // Calculate the movement
        float dx_est = dis * Mathf.Cos(theta_r_est);
        float dy_est = dis * Mathf.Sin(theta_r_est);
        // The expected robot states
        X_true.SetSubVector(0, 3, Vector<float>.Build.DenseOfArray(new float[] { X_true[0] + dx_est, X_true[1] + dy_est, theta_r_est }));

        return X_true;
    }

    // Only update robots
    Vector<float> MotionUpdateActual(Vector<float> X_actual, float dis, float ang)
    {
        float dis_noise = Mathf.Pow(0.02f, 2) * (float)normal.Sample();
        float ang_noise = Mathf.Pow(Mathf.Deg2Rad * 1f, 2) * (float)normal.Sample();

        float theta_r_actual = ClampRad(ang + X_actual[2]);
        // Calculate the movement
        float dx_actual = (dis + dis_noise) * Mathf.Cos(theta_r_actual);
        float dy_actual = (dis + dis_noise) * Mathf.Sin(theta_r_actual);
        theta_r_actual = ClampRad(theta_r_actual + ang_noise);

        X_actual = Vector<float>.Build.DenseOfArray(new float[] { X_actual[0] + dx_actual, X_actual[1] + dy_actual, theta_r_actual });

        return X_actual;
    }


    Vector<float> ObervationEstimate(Vector<float> X_est, Vector<float> X_lm_est)
    {
        Vector<float> d = X_est.SubVector(0, 2) - X_lm_est;
        float r = (float)d.L2Norm();

        float b = Mathf.Atan2(X_lm_est[1] - X_est[1], X_lm_est[0] - X_est[0]) - X_est[2];
        b = ClampRad(b);

        return Vector<float>.Build.DenseOfArray(new float[] { r, b });
    }

    Dictionary<int, Vector<float>> ObserveLandmarks(Vector<float> X_true)
    {
        float x_r = X_true[0];
        float y_r = X_true[1];
        float theta_r = X_true[2];
        Dictionary<int, Vector<float>> observedLandmarks = new Dictionary<int, Vector<float>>();

        foreach (int idx in landmarkDictionary.Keys)
        {
            Vector<float> locationVector = landmarkDictionary[idx];
            float x_m = locationVector[0];
            float y_m = locationVector[1];

            //range
            float r = Mathf.Sqrt(Mathf.Pow(x_m - x_r, 2) + Mathf.Pow(y_m - y_r, 2));
            r += Mathf.Pow(0.2f, 2) * (float)normal.Sample();

            //bearing, angle(arctan - theta)
            float b = Mathf.Atan2(y_m - y_r, x_m - x_r) - theta_r;
            b += Mathf.Pow(Mathf.Deg2Rad * 5.0f, 2) * (float)normal.Sample();
            b = ClampRad(b);

            Vector<float> measurementVector = Vector<float>.Build.DenseOfArray(new float[] { r, b });

            if (Mathf.Abs(b) <= measureRange[2] && r >= measureRange[0] && r <= measureRange[1])
            {
                observedLandmarks.Add(idx, measurementVector);
            }
        }
        return observedLandmarks;
    }

    float ClampRad(float ang)
    {
        float new_ang = ang;

        if (ang > Mathf.PI)
            new_ang -= 2 * Mathf.PI;
        else if (ang < -Mathf.PI)
            new_ang += 2 * Mathf.PI;

        return new_ang;
    }

    List<int[]> GetIndexPairs(int length)
    {
        List<int[]> pairList = new List<int[]>();

        for (int i = 0; i < length; i++)
            for (int j = i + 1; j < length; j++)
                pairList.Add(new int[2] { i, j });

        return pairList;
    }
}
