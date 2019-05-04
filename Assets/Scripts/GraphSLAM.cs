using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Complex.Solvers;

public class GraphSLAM : MonoBehaviour
{
    [SerializeField]
    int recordInterval = 5;

    int count = 0;

    [SerializeField]
    int optimizeEpoch = 20;

    public int historyLength = 100;

    bool isOptimizing = false;

    Queue<Vector<float>> robotTrace_predict = new Queue<Vector<float>>();
    Queue<Vector<float>> robotTrace_actual = new Queue<Vector<float>>();
    Queue<Dictionary<int, Vector<float>>> landmarkTrace = new Queue<Dictionary<int, Vector<float>>>();

    //States vector
    Vector<float> X_rob_predict, X_rob_actual;

    float transNoiseFactor = 0.2f, angNoiseFactor = 5f;
    float rangeNoiseFactor = 0.02f, bearingNoiseFactor = 1.5f;

    Normal normal = Normal.WithMeanPrecision(0, 1);
    Matrix<float> R, Q;

    Dictionary<int, Vector<float>> observedLandmarks = new Dictionary<int, Vector<float>>();

    float targetDis = 0f, targetAng = 0f, absAng = 0f;
    //bool isTargetReached = false;

    float threshold = 0.01f;

    //Car parameters
    [SerializeField]
    float transVelocity = 10f;

    [SerializeField]
    float rotVelocity = 10f;

    [SerializeField]
    float rangeMin = 0.1f;

    [SerializeField]
    float rangeMax = 1.5f;

    [SerializeField]
    float bearing = Mathf.PI / 4;

    int robotLandmarkIndex = 0;
    // Dictionary that stores associated landmarks
    public Dictionary<int, Vector<float>> landmarkDictionary =
        new Dictionary<int, Vector<float>>();

    GraphVisualizer graphVisualizer;

    private void Awake()
    {
        graphVisualizer = GetComponent<GraphVisualizer>();
    }
    // Start is called before the first frame update
    void Start()
    {
        R = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(rangeNoiseFactor, 2), Mathf.Pow(Mathf.Deg2Rad * bearingNoiseFactor, 2) });
        Q = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(Mathf.Deg2Rad * angNoiseFactor, 2) });
        InitializeRobotState(0f, 0f, 0f);
    }

    void FixedUpdate()
    {
        if (isOptimizing)
            return;


        if (absAng > 0)
        {
            float delta_ang = Time.fixedDeltaTime * rotVelocity;
            Graph_Update(0f, Mathf.Sign(targetAng) * delta_ang);
            absAng -= delta_ang;
        }
        else
        {
            if (targetDis > 0)
            {
                float delta_dis = Time.fixedDeltaTime * transVelocity;
                Graph_Update(delta_dis, 0f);
                targetDis -= delta_dis;
            }
        }
    }

    public void SetTargetWorldPoints(float world_x, float world_y)
    {
        CalculateTargetPose(X_rob_actual, world_x, world_y, out targetDis, out targetAng, out absAng);
    }

    void Graph_Update(float deltaDis, float deltaAng)
    {
        //Debug.Log("Predict count: " + robotTrace_predict.Count);
        //Debug.Log("Actual count: " + robotTrace_actual.Count);

        //Shall not happen
        if (robotTrace_predict.Count != robotTrace_actual.Count)
            return;

        //Debug.Log("Graph update");

        //Avoid Overflow
        if (robotTrace_predict.Count >= historyLength)
        {
            //Dequeue
            robotTrace_predict.Dequeue();
            robotTrace_actual.Dequeue();
        }

        float ang_noise = deltaDis / 1 * Mathf.Deg2Rad * angNoiseFactor * (float)normal.Sample(),
            dis_noise = deltaDis / 1 * transNoiseFactor * (float)normal.Sample();

        StatesUpdate(ref X_rob_actual, deltaDis + dis_noise, deltaAng + ang_noise);
        StatesUpdate(ref X_rob_predict, deltaDis, deltaAng);
        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);

        if (deltaDis != 0)
        {
            if (count == recordInterval)
            {
                graphVisualizer.Visualize(X_rob_predict, X_rob_actual, true);
                landmarkTrace.Enqueue(observedLandmarks);
                robotTrace_predict.Enqueue(X_rob_predict.Clone());
                robotTrace_actual.Enqueue(X_rob_actual.Clone());

                count = 0;
            }
            else
                graphVisualizer.Visualize(X_rob_predict, X_rob_actual, false);

            count++;
        }
    }

    public void Optimize()
    {
        Debug.Log("Optimize");

        isOptimizing = true;

        for (int i = 0; i < optimizeEpoch; i++)
        {
            OptimizeOnce(ref X_rob_predict, ref robotTrace_predict);
        }

        isOptimizing = !graphVisualizer.OptimizedVisualization(robotTrace_predict);
    }

    void OptimizeOnce(ref Vector<float> latest_X_predict, ref Queue<Vector<float>> robotTrace_predict)
    {
        bool containSameLandmarks = false;

        Dictionary<int, Vector<float>>[] landmarkTrace_array = landmarkTrace.ToArray();
        Vector<float> robotTrace_predict_vector = QueueToVector(robotTrace_predict);

        Vector<float> b = Vector<float>.Build.SparseOfArray(new float[robotTrace_predict_vector.Count]);
        Matrix<float> H = Matrix<float>.Build.SparseOfArray(new float[robotTrace_predict_vector.Count, robotTrace_predict_vector.Count]);

        int[] flagArray = new int[robotTrace_predict_vector.Count / 3];

        //For each combination
        foreach (int[] indexPair in GetIndexPairs(robotTrace_predict_vector.Count / 3))
        {

            Vector<float> X_i = robotTrace_predict_vector.SubVector(3 * indexPair[0], 3);
            Dictionary<int, Vector<float>> lmObs_i = landmarkTrace_array[indexPair[0]];

            Vector<float> X_j = robotTrace_predict_vector.SubVector(3 * indexPair[1], 3);
            Dictionary<int, Vector<float>> lmObs_j = landmarkTrace_array[indexPair[1]];

            //Debug.Log("X_i observed landmarks: " + lmObs_i.Count);
            //Debug.Log("X_j observed landmarks: " + lmObs_j.Count);


            Matrix<float> H_ii = H.SubMatrix(3 * indexPair[0], 3, 3 * indexPair[0], 3);
            Matrix<float> H_ij = H.SubMatrix(3 * indexPair[0], 3, 3 * indexPair[1], 3);
            Matrix<float> H_ji = H.SubMatrix(3 * indexPair[1], 3, 3 * indexPair[0], 3);
            Matrix<float> H_jj = H.SubMatrix(3 * indexPair[1], 3, 3 * indexPair[1], 3);

            Vector<float> b_i = b.SubVector(3 * indexPair[0], 3);
            Vector<float> b_j = b.SubVector(3 * indexPair[1], 3);

            Matrix<float> omega = GetOmega(X_i, X_j);

            //Loop through all landmarks
            foreach (int lmIdx in landmarkDictionary.Keys)
            {
                //If same landmark is observed at different robot pos
                if (lmObs_i.ContainsKey(lmIdx) && lmObs_j.ContainsKey(lmIdx))
                {
                    if (!containSameLandmarks)
                        containSameLandmarks = true;

                    //set flag
                    flagArray[indexPair[0]] = 1;
                    flagArray[indexPair[1]] = 1;

                    Vector<float> err = GetError(X_i, X_j, lmObs_i[lmIdx], lmObs_j[lmIdx]);
                    Matrix<float> A = GetJacobianA(X_i, lmObs_i[lmIdx]);
                    Matrix<float> B = -GetJacobianA(X_j, lmObs_j[lmIdx]);
                    
                    //Omega, update H, b

                    H_ii += A.Transpose() * omega * A;
                    H_ij += A.Transpose() * omega * B;
                    H_ji += B.Transpose() * omega * A;
                    H_jj += B.Transpose() * omega * B;

                    b_i += A.Transpose() * omega * err;
                    b_j += B.Transpose() * omega * err;
                }
            }

            H.SetSubMatrix(3 * indexPair[0], 3, 3 * indexPair[0], 3, H_ii);
            H.SetSubMatrix(3 * indexPair[0], 3, 3 * indexPair[1], 3, H_ij);
            H.SetSubMatrix(3 * indexPair[1], 3, 3 * indexPair[0], 3, H_ji);
            H.SetSubMatrix(3 * indexPair[1], 3, 3 * indexPair[1], 3, H_jj);

            b.SetSubVector(3 * indexPair[0], 3, b_i);
            b.SetSubVector(3 * indexPair[1], 3, b_j);
        }

        if (containSameLandmarks)
        {
            //Debug.Log("Solve the equation...");

            //Fill in identity matrix
            //for (int i = 0; i < flagArray.Length; i++)
            //{
            //    if (flagArray[i] == 0)
            //    {
            //        Matrix<float> H_kk = H.SubMatrix(3 * i, 3, 3 * i, 3);
            //        H_kk += Matrix<float>.Build.SparseIdentity(3);
            //        H.SetSubMatrix(3 * i, 3, 3 * i, 3, H_kk);
            //    }
            //}


            //Matrix<float> H_11 = H.SubMatrix(0, 3, 0, 3);
            //H_11 += Matrix<float>.Build.SparseIdentity(3);
            //H.SetSubMatrix(0, 3, 0, 3, H_11);



            Debug.Log(H.RowCount);
            Debug.Log(H.Rank());


            var svd = H.Svd(true);
            Debug.Log(svd.Rank);
            Debug.Log(svd.W.RowCount);

            //Debug.Log(H.ToString());
            Debug.Log("b: ");
            Debug.Log(b.ToString());

            //Solve
            Vector<float> X_delta = H.Solve(-b);

            Debug.Log("X_delta: ");
            Debug.Log(X_delta.ToString());

            for (int i = 0; i < robotTrace_predict_vector.Count; i++)
            {
                if (!float.IsNaN(X_delta[i]) && !float.IsInfinity(X_delta[i]))
                {
                    robotTrace_predict_vector[i] += X_delta[i];
                }
            }

            //robotTrace_predict_vector += X_delta;

            latest_X_predict[0] = robotTrace_predict_vector[robotTrace_predict_vector.Count - 3];
            latest_X_predict[1] = robotTrace_predict_vector[robotTrace_predict_vector.Count - 2];
            latest_X_predict[2] = robotTrace_predict_vector[robotTrace_predict_vector.Count - 1];
        }

        robotTrace_predict = VectorToQueue(robotTrace_predict_vector);
    }

    Matrix<float> GetOmega(Vector<float> X_i, Vector<float> X_j/*, Vector<float> z_i, Vector<float> z_j*/)
    {
        Matrix<float> rt_i = GetRotationMatrix(X_i[2]);
        Matrix<float> rt_j = GetRotationMatrix(X_j[2]);

        return (rt_i * Q * rt_i.Transpose() + rt_j * Q * rt_j.Transpose()).Inverse();

        //return Matrix<float>.Build.DenseIdentity(3);

        //float ang_j = ClampRad(X_j[2] + z_j[1]);
        //float ang_i = ClampRad(X_i[2] + z_i[1]);

        //Vector<float> X_delta_i = Vector<float>.Build.DenseOfArray(new float[3]
        //{
        //    X_i[0] + z_i[0] * Mathf.Cos(ang_i),
        //    X_i[1] + z_i[0] * Mathf.Sin(ang_i),
        //    ang_i
        //});

        //Vector<float> X_delta_j = Vector<float>.Build.DenseOfArray(new float[3]
        //{
        //    X_j[0] + z_j[0] * Mathf.Cos(ang_j),
        //    X_j[1] + z_j[0] * Mathf.Sin(ang_j),
        //    ang_j
        //});

        //Vector<float> diff = X_delta_j - X_delta_i;
        //Matrix<float> diff_norm = (diff - diff.Sum()).ToRowMatrix();

        //Debug.Log((diff_norm.Transpose() * diff_norm).Inverse());

        //return (diff_norm.Transpose() * diff_norm).Inverse();
    }

    Matrix<float> GetRotationMatrix(float theta)
    {
        return Matrix<float>.Build.DenseOfArray(new float[3, 3]
            {{Mathf.Cos(theta), -Mathf.Sin(theta), 0 }, 
            {Mathf.Sin(theta), Mathf.Cos(theta), 0}, 
            {0, 0, 1.0f }});
    }


    Vector<float> GetError(Vector<float> X_i, Vector<float> X_j, Vector<float> z_i, Vector<float> z_j)
    {
        float ang_j = ClampRad(X_j[2] + z_j[1]);
        float ang_i = ClampRad(X_i[2] + z_i[1]);


        float e_x = X_j[0] + z_j[0] * Mathf.Cos(ang_j) - (X_i[0] + z_i[0] * Mathf.Cos(ang_i));
        float e_y = X_j[1] + z_j[0] * Mathf.Sin(ang_j) - (X_i[1] + z_i[0] * Mathf.Sin(ang_i));
        float e_a = ang_j - ang_i;

        return Vector<float>.Build.DenseOfArray(new float[3] { e_x, e_y, e_a });
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
        X_rob_predict = Vector<float>.Build.DenseOfArray(new float[] { 0, 0, 0 });
        X_rob_actual = X_rob_predict.Clone();

        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);

        robotTrace_predict.Enqueue(X_rob_predict.Clone());
        robotTrace_actual.Enqueue(X_rob_actual.Clone());
        landmarkTrace.Enqueue(observedLandmarks);
        graphVisualizer.VisualizerRobotRegistration(x, y, theta);
    }

    public void InitializeLandmarks(int numOfLandmarks, float worldLimitX, float worldLimitY)
    {
        landmarkDictionary.Clear();

        //landmarkDictionary.Add(0, Vector<float>.Build.DenseOfArray(new float[] { 0.5f, 0.5f }));
        //graphVisualizer.VisualizerLandmarkRegistration(0, 0.5f, 0.5f);

        for (int i = 0; i < numOfLandmarks; i++)
        {
            float x = UnityEngine.Random.Range(0, worldLimitX);
            float y = UnityEngine.Random.Range(0, worldLimitY);

            landmarkDictionary.Add(i, Vector<float>.Build.DenseOfArray(new float[] { x, y }));
            graphVisualizer.VisualizerLandmarkRegistration(i, x, y);
        }
    }

    void CalculateTargetPose(Vector<float> X_true, float x_world, float y_world, out float t_dis, out float t_ang, out float abs_ang)
    {
        float x_r = X_true[0];
        float y_r = X_true[1];
        float theta_r = X_true[2];

        // Calculate distance and theta from target world point
        t_dis = Mathf.Sqrt(Mathf.Pow(x_world - x_r, 2) + Mathf.Pow(y_world - y_r, 2));
        t_ang = Mathf.Atan2(y_world - y_r, x_world - x_r) - theta_r;
        t_ang = ClampRad(t_ang);

        abs_ang = Mathf.Abs(t_ang);
    }

    void StatesUpdate(ref Vector<float> currentState, float dis, float ang)
    {
        float theta_r = ClampRad(ang + currentState[2]);
        // Calculate the movement
        float dx = dis * Mathf.Cos(theta_r);
        float dy = dis * Mathf.Sin(theta_r);
        currentState.SetSubVector(0, 3, Vector<float>.Build.DenseOfArray(new float[] { currentState[0] + dx, currentState[1] + dy, theta_r }));
    }

    Vector<float> ObervationEstimate(Vector<float> X_est, Vector<float> X_lm_est)
    {
        CalculateMovement(out float r, out float b, X_est, X_lm_est);
        return Vector<float>.Build.DenseOfArray(new float[] { r, b });
    }

    Dictionary<int, Vector<float>> ObserveLandmarks(Vector<float> X_true, float rangeNoiseFactor, float bearingNoiseFactor)
    {
        Dictionary<int, Vector<float>> observations = new Dictionary<int, Vector<float>>();

        foreach (int idx in landmarkDictionary.Keys)
        {
            CalculateMovement(out float r, out float b, X_true, landmarkDictionary[idx]);

            float bearing_noise = Mathf.Deg2Rad * bearingNoiseFactor * (float)normal.Sample(),
              range_noise = rangeNoiseFactor * (float)normal.Sample();

            r += range_noise;
            b = ClampRad(b + bearing_noise);

            if (Mathf.Abs(b) <= bearing && r >= rangeMin && r <= rangeMax)
            {
                observations.Add(idx, Vector<float>.Build.DenseOfArray(new float[] { r, b }));
            }
        }

        return observations;
    }

    void CalculateMovement(out float d, out float a, Vector<float> currentState, Vector<float> targetPos)
    {
        d = (float)(currentState.SubVector(0, 2) - targetPos).L2Norm();
        a = ClampRad(Mathf.Atan2(targetPos[1] - currentState[1], targetPos[0] - currentState[0]) - currentState[2]);
    }

    List<int[]> GetIndexPairs(int length)
    {
        List<int[]> pairList = new List<int[]>();

        for (int i = 0; i < length; i++)
            for (int j = i + 1; j < length; j++)
                pairList.Add(new int[2] { i, j });

        //for (int i = 0; i < length-1; i++)
        //    pairList.Add(new int[2] { i, length - 1 });

        return pairList;
    }

    #region Utility

    float ClampRad(float ang)
    {
        float new_ang = ang;

        if (ang > Mathf.PI)
            new_ang -= 2 * Mathf.PI;
        else if (ang < -Mathf.PI)
            new_ang += 2 * Mathf.PI;

        return new_ang;
    }



    Vector<float> QueueToVector(Queue<Vector<float>> robotTrace_predict)
    {
        Vector<float> robStateVector = Vector<float>.Build.DenseOfArray(new float[robotTrace_predict.Count * 3]);

        int count = 0;

        while (robotTrace_predict.Count > 0)
        {
            Vector<float> state = robotTrace_predict.Dequeue();
            robStateVector.SetSubVector(3 * count, 3, state);
            count++;
        }

        return robStateVector;
    }

    Queue<Vector<float>> VectorToQueue(Vector<float> robStateVector)
    {
        Queue<Vector<float>> robotTrace_actual = new Queue<Vector<float>>();

        for (int i = 0; i < robStateVector.Count / 3; i++)
        {
            robotTrace_actual.Enqueue(robStateVector.SubVector(3 * i, 3));
        }

        return robotTrace_actual;
    }


    #endregion
}
