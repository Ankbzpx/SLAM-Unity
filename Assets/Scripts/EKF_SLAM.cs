using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

[RequireComponent(typeof(EKF_Visualizer))]
public class EKF_SLAM : MonoBehaviour
{
    //States vector
    Vector<float> X, X_rob_actual;
    //Covariance
    Matrix<float> P;

    public float transNoiseFactor = 0.1f, angNoiseFactor = 5f;
    public float rangeNoiseFactor = 0.01f, bearingNoiseFactor = 1f;

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

    EKF_Visualizer visualizer;
    int robotLandmarkIndex = 0;
    // Dictionary that stores associated landmarks
    public Dictionary<int, Vector<float>> landmarkExternalDictionary =
        new Dictionary<int, Vector<float>>();

    // Input world landmark index, output local landmark index
    public Dictionary<int, int> worldToLocalDictionary =
        new Dictionary<int, int>();

    private void Awake()
    {
        visualizer = GetComponent<EKF_Visualizer>();
    }
    // Start is called before the first frame update
    void Start()
    {
        R = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { rangeNoiseFactor, Mathf.Deg2Rad * bearingNoiseFactor });
        Q = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(transNoiseFactor, 2), Mathf.Pow(Mathf.Deg2Rad * angNoiseFactor, 2) });
        InitializeRobotState(0f, 0f, 0f);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (absAng > 0)
        {
            float delta_ang = Time.fixedDeltaTime * rotVelocity;
            EKF_Update(0f, Mathf.Sign(targetAng)* delta_ang);

            absAng -= delta_ang;
        }
        else
        {
            if (targetDis > 0)
            {
                float delta_dis = Time.fixedDeltaTime * transVelocity;
                EKF_Update(delta_dis, 0f);
                targetDis -= delta_dis;
            }
        }
    }

    public void SetTargetWorldPoints(float world_x, float world_y)
    {
        CalculateTargetPose(X_rob_actual, world_x, world_y, out targetDis, out targetAng, out absAng);
    }

    void EKF_Update(float deltaDis, float deltaAng)
    {
        float ang_noise = deltaDis / 1 * Mathf.Deg2Rad*angNoiseFactor * (float)normal.Sample(),
            dis_noise = deltaDis / 1* transNoiseFactor * (float)normal.Sample();


        StatesUpdate(ref X_rob_actual, deltaDis + dis_noise, deltaAng + ang_noise);
        StatesUpdate(ref X, deltaDis, deltaAng);
        CovarianceUpdate(ref P, X, deltaDis, X[2]);

        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);
        ObservationUpdate(ref X, ref P, X_rob_actual, observedLandmarks);

        visualizer.Visualize(X, X_rob_actual, P);
        visualizer.VisualizeMeasurement(observedLandmarks);
    }

    void InitializeRobotState(float x, float y, float theta)
    {
        X = Vector<float>.Build.DenseOfArray(new float[] { 0, 0, 0 });
        P = Matrix<float>.Build.DenseOfDiagonalArray(new float[] { 1, 1, 0.01f });
        P = 0.5f * (P + P.Transpose());
        X_rob_actual = X.Clone();

        visualizer.VisualizerRobotRegistration(x, y, theta);
    }

    public void InitializeLandmarks(int numOfLandmarks, float worldLimitX, float worldLimitY)
    {
        landmarkExternalDictionary.Clear();

        landmarkExternalDictionary.Add(0, Vector<float>.Build.DenseOfArray(new float[] { 0.5f, 0.5f }));
        visualizer.VisualizeTrueLandmarks(0, 0.5f, 0.5f);

        for (int i = 1; i < numOfLandmarks; i++)
        {
            float x = Random.Range(0, worldLimitX);
            float y = Random.Range(0, worldLimitY);

            landmarkExternalDictionary.Add(i, Vector<float>.Build.DenseOfArray(new float[] { x, y }));
            visualizer.VisualizeTrueLandmarks(i, x, y);
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

    void CovarianceUpdate(ref Matrix<float> P_est, Vector<float> X_est, float dis, float theta_est)
    {
        int n = P_est.RowCount;
        Matrix<float> J1 = Matrix<float>.Build.DenseOfArray(new float[3, 3] { { 1, 0, -dis * Mathf.Sin(theta_est) }, { 0, 1, dis * Mathf.Cos(theta_est) }, { 0, 0, 1 } });
        Matrix<float> J2 = Matrix<float>.Build.DenseOfArray(new float[3, 3] { { Mathf.Cos(X_est[2]), -Mathf.Sin(X_est[2]), 0 }, { Mathf.Sin(X_est[2]), Mathf.Cos(X_est[2]), 0 }, { 0, 0, 1 } });

        if (n > 3)
        {
            Matrix<float> PRM = J1 * P_est.SubMatrix(0, 3, 3, n - 3);
            P_est.SetSubMatrix(0, 0, J1 * P_est.SubMatrix(0, 3, 0, 3) * J1.Transpose() + J2 * Q * J2.Transpose());
            P_est.SetSubMatrix(0, 3, PRM);
            P_est.SetSubMatrix(3, 0, PRM.Transpose());
        }   
    }

    Vector<float> ObervationEstimate(Vector<float> X_est, Vector<float> X_lm_est)
    {
        CalculateMovement(out float r, out float b, X_est, X_lm_est);
        return Vector<float>.Build.DenseOfArray(new float[] { r, b });
    }

    Matrix<float> LmJacobian(Vector<float> X_est, Vector<float> X_lm_est)
    {
        Vector<float> d = X_est.SubVector(0, 2) - X_lm_est;
        float r = (float)d.L2Norm();

        return Matrix<float>.Build.DenseOfArray(new float[2, 3]
            {{(d[0])/r, (d[1])/r, 0},
            {(-d[1])/Mathf.Pow(r, 2), (d[0])/Mathf.Pow(r, 2), -1 }});
    }

    Dictionary<int, Vector<float>> ObserveLandmarks(Vector<float> X_true, float rangeNoiseFactor, float bearingNoiseFactor)
    {
        Dictionary<int, Vector<float>>  observations = new Dictionary<int, Vector<float>>();

        foreach (int idx in landmarkExternalDictionary.Keys)
        {
            CalculateMovement(out float r, out float b, X_true, landmarkExternalDictionary[idx]);

            float bearing_noise = Mathf.Deg2Rad*bearingNoiseFactor * (float)normal.Sample(),
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

    // Update both robot and landmarks
    void ObservationUpdate(ref Vector<float> X, ref Matrix<float> P, Vector<float> X_rob_actual, Dictionary<int, Vector<float>> observedLandmarks)
    {
        foreach (var landmark in observedLandmarks)
        {
            int world_id = landmark.Key;
            Vector<float> z_measure = landmark.Value;


            //Observation Update
            float r_measure = z_measure[0];
            float b_measure = z_measure[1];

            //If it is a new landmark
            if (!worldToLocalDictionary.ContainsKey(world_id))
            {

                // Register the new landmark
                worldToLocalDictionary.Add(world_id, robotLandmarkIndex);
                visualizer.VisualizerLandmarkRegistration(robotLandmarkIndex);
                robotLandmarkIndex++;

                //Debug.Log("New landmarks: " + world_id);

                int n_states = X.Count;

                //Estimate landmark based on estimate states with true measurement
                float x_r = X[0];
                float y_r = X[1];
                float theta_r = X[2];
                float alpha = ClampRad(theta_r + b_measure);

                Vector<float> X_lm_new = Vector<float>.Build.DenseOfArray(new float[]
                {x_r + r_measure*Mathf.Cos(alpha),
                    y_r + r_measure*Mathf.Sin(alpha) });

                Vector<float> X_new = Vector<float>.Build.DenseOfArray(new float[n_states + 2]);
                X_new.SetSubVector(0, n_states, X);
                X_new.SetSubVector(n_states, 2, X_lm_new);

                X = X_new; 

                Matrix<float> Jxr = Matrix<float>.Build.DenseOfArray(new float[2, 3]
                {{1, 0, -r_measure*Mathf.Sin(alpha)},
                {0, 1, r_measure*Mathf.Cos(alpha)}});

                Matrix<float> Jz = Matrix<float>.Build.DenseOfArray(new float[2, 2]
                {{Mathf.Cos(alpha), -r_measure*Mathf.Sin(alpha)},
                {Mathf.Sin(alpha), r_measure*Mathf.Cos(alpha)}});

                Matrix<float> Pvv = P.SubMatrix(0, 3, 0, 3);
                //Debug.Log("Pvv" + Pvv.ToString());

                Matrix<float> right_top = Pvv * Jxr.Transpose();
                Matrix<float> right_buttom = Jxr * Pvv * Jxr.Transpose() + Jz * R * Jz.Transpose();
                //Matrix<float> right_buttom = Matrix<float>.Build.DenseIdentity(2, 2) * 1000;

                P = P.DiagonalStack(right_buttom);

                P.SetSubMatrix(0, n_states, right_top);
                P.SetSubMatrix(n_states, 0, right_top.Transpose());

                if (n_states > 3)
                {
                    Matrix<float> Pmv = P.SubMatrix(3, n_states - 3, 0, 3);
                    Matrix<float> right_middle = Pmv * Jxr.Transpose();

                    P.SetSubMatrix(3, n_states, right_middle);
                    P.SetSubMatrix(n_states, 3, right_middle.Transpose());
                }

                P = 0.5f * (P + P.Transpose());
            }

            //Debug.Log("Old landmark re-observed: " + world_id);
            int local_id = worldToLocalDictionary[world_id];
            int index = 3 + local_id * 2;

            //Estimate measurement based on estimate robot states and estimate landmark states 
            Vector<float> z_est = ObervationEstimate(X, X.SubVector(index, 2));
            Matrix<float> h = LmJacobian(X, X.SubVector(index, 2));

            Matrix<float> H = Matrix<float>.Build.DenseOfArray(new float[2, X.Count]);
            H.SetSubMatrix(0, 0, h);
            H.SetSubMatrix(0, index, -h.SubMatrix(0, 2, 0, 2));

            Matrix<float> inov = H * P * H.Transpose() + R;

            //Kalman gain
            Matrix<float> K = P * H.Transpose() * inov.Inverse();
            Vector<float> diff = z_measure - z_est;
            diff[1] = ClampRad(diff[1]);

            Vector<float> X_inc = K * diff;

            X += X_inc;
            X[2] = ClampRad(X[2]);

            P = (Matrix<float>.Build.DenseIdentity(X.Count, X.Count) - K * H) * P;
        }
    }

    public void ResetAll()
    {
        robotLandmarkIndex = 0;
        landmarkExternalDictionary.Clear();
        worldToLocalDictionary.Clear();
        InitializeRobotState(0f, 0f, 0f);
        targetDis = 0f;
        targetAng = 0f;
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
}
