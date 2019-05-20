using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System;
using Random = System.Random;

[RequireComponent(typeof(EKF_Visualizer))]
public class EKF_SLAM : MonoBehaviour
{
    //States vector
    Vector<double> X, X_rob_actual;
    //Covariance
    Matrix<double> P;

    public double transNoiseFactor = 0.1, angNoiseFactor = 5;
    public double rangeNoiseFactor = 0.01, bearingNoiseFactor = 1;

    Normal normal = Normal.WithMeanPrecision(0, 1);
    Matrix<double> R, Q;

    Dictionary<int, Vector<double>> observedLandmarks = new Dictionary<int, Vector<double>>();

    double targetDis = 0, targetAng = 0, absAng = 0;
    //bool isTargetReached = false;

    double threshold = 0.01;

    Random random;

    //Car parameters
    [SerializeField]
    double transVelocity = 10;

    [SerializeField]
    double rotVelocity = 10;

    [SerializeField]
    double rangeMin = 0.1;

    [SerializeField]
    double rangeMax = 1.5;

    [SerializeField]
    double bearing = Math.PI / 4;

    EKF_Visualizer visualizer;
    int robotLandmarkIndex = 0;

    const double Deg2Rad = Math.PI /180;

    // Dictionary that stores associated landmarks
    public Dictionary<int, Vector<double>> landmarkExternalDictionary =
        new Dictionary<int, Vector<double>>();

    // Input world landmark index, output local landmark index
    public Dictionary<int, int> worldToLocalDictionary =
        new Dictionary<int, int>();

    private void Awake()
    {
        random = new Random();
        visualizer = GetComponent<EKF_Visualizer>();
    }
    // Start is called before the first frame update
    void Start()
    {
        R = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { rangeNoiseFactor, Deg2Rad * bearingNoiseFactor });
        Q = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { Math.Pow(transNoiseFactor, 2), Math.Pow(transNoiseFactor, 2), Math.Pow(Deg2Rad * angNoiseFactor, 2) });
        InitializeRobotState(0, 0, 0);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (absAng > 0)
        {
            double delta_ang = Time.fixedDeltaTime * rotVelocity;
            EKF_Update(0, Math.Sign(targetAng)* delta_ang);

            absAng -= delta_ang;
        }
        else
        {
            if (targetDis > 0)
            {
                double delta_dis = Time.fixedDeltaTime * transVelocity;
                EKF_Update(delta_dis, 0);
                targetDis -= delta_dis;
            }
        }
    }

    public void SetTargetWorldPoints(double world_x, double world_y)
    {
        CalculateTargetPose(X_rob_actual, world_x, world_y, out targetDis, out targetAng, out absAng);
    }

    void EKF_Update(double deltaDis, double deltaAng)
    {
        double ang_noise = deltaDis / 1 * Deg2Rad*angNoiseFactor * normal.Sample(),
            dis_noise = deltaDis / 1* transNoiseFactor * (double)normal.Sample();


        StatesUpdate(ref X_rob_actual, deltaDis + dis_noise, deltaAng + ang_noise);
        StatesUpdate(ref X, deltaDis, deltaAng);
        CovarianceUpdate(ref P, X, deltaDis, X[2]);

        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);
        ObservationUpdate(ref X, ref P, X_rob_actual, observedLandmarks);

        visualizer.Visualize(X, X_rob_actual, P);
        visualizer.VisualizeMeasurement(observedLandmarks);
    }

    void InitializeRobotState(double x, double y, double theta)
    {
        X = Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0 });
        P = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { 1, 1, 0.01 });
        P = 0.5 * (P + P.Transpose());
        X_rob_actual = X.Clone();

        visualizer.VisualizerRobotRegistration((float)x, (float)y, (float)theta);
    }

    public void InitializeLandmarks(int numOfLandmarks, double worldLimitX, double worldLimitY)
    {
        landmarkExternalDictionary.Clear();

        landmarkExternalDictionary.Add(0, Vector<double>.Build.DenseOfArray(new double[] { 0.5, 0.5}));
        visualizer.VisualizeTrueLandmarks(0, 0.5f, 0.5f);

        for (int i = 1; i < numOfLandmarks; i++)
        {
            double x = random.NextDouble() * worldLimitX;
            double y = random.NextDouble() * worldLimitY;

            landmarkExternalDictionary.Add(i, Vector<double>.Build.DenseOfArray(new double[] { x, y }));
            visualizer.VisualizeTrueLandmarks(i, (float)x, (float)y);
        }
    }

    void CalculateTargetPose(Vector<double> X_true, double x_world, double y_world, out double t_dis, out double t_ang, out double abs_ang)
    {
        double x_r = X_true[0];
        double y_r = X_true[1];
        double theta_r = X_true[2];

        // Calculate distance and theta from target world point
        t_dis = Math.Sqrt(Math.Pow(x_world - x_r, 2) + Math.Pow(y_world - y_r, 2));
        t_ang = Math.Atan2(y_world - y_r, x_world - x_r) - theta_r;
        t_ang = ClampRad(t_ang);

        abs_ang = Math.Abs(t_ang);
    }

    void StatesUpdate(ref Vector<double> currentState, double dis, double ang)
    {
        double theta_r = ClampRad(ang + currentState[2]);
        // Calculate the movement
        double dx = dis * Math.Cos(theta_r);
        double dy = dis * Math.Sin(theta_r);
        currentState.SetSubVector(0, 3, Vector<double>.Build.DenseOfArray(new double[] { currentState[0] + dx, currentState[1] + dy, theta_r }));
    }

    void CovarianceUpdate(ref Matrix<double> P_est, Vector<double> X_est, double dis, double theta_est)
    {
        int n = P_est.RowCount;
        Matrix<double> J1 = Matrix<double>.Build.DenseOfArray(new double[3, 3] { { 1, 0, -dis * Math.Sin(theta_est) }, { 0, 1, dis * Math.Cos(theta_est) }, { 0, 0, 1 } });
        Matrix<double> J2 = Matrix<double>.Build.DenseOfArray(new double[3, 3] { { Math.Cos(X_est[2]), -Math.Sin(X_est[2]), 0 }, { Math.Sin(X_est[2]), Math.Cos(X_est[2]), 0 }, { 0, 0, 1 } });

        if (n > 3)
        {
            Matrix<double> PRM = J1 * P_est.SubMatrix(0, 3, 3, n - 3);
            P_est.SetSubMatrix(0, 0, J1 * P_est.SubMatrix(0, 3, 0, 3) * J1.Transpose() + J2 * Q * J2.Transpose());
            P_est.SetSubMatrix(0, 3, PRM);
            P_est.SetSubMatrix(3, 0, PRM.Transpose());
        }   
    }

    Vector<double> ObervationEstimate(Vector<double> X_est, Vector<double> X_lm_est)
    {
        CalculateMovement(out double r, out double b, X_est, X_lm_est);
        return Vector<double>.Build.DenseOfArray(new double[] { r, b });
    }

    Matrix<double> LmJacobian(Vector<double> X_est, Vector<double> X_lm_est)
    {
        Vector<double> d = X_est.SubVector(0, 2) - X_lm_est;
        double r = d.L2Norm();

        return Matrix<double>.Build.DenseOfArray(new double[2, 3]
            {{(d[0])/r, (d[1])/r, 0},
            {(-d[1])/Math.Pow(r, 2), (d[0])/Math.Pow(r, 2), -1 }});
    }

    Dictionary<int, Vector<double>> ObserveLandmarks(Vector<double> X_true, double rangeNoiseFactor, double bearingNoiseFactor)
    {
        Dictionary<int, Vector<double>> observations = new Dictionary<int, Vector<double>>();

        foreach (int idx in landmarkExternalDictionary.Keys)
        {
            CalculateMovement(out double r, out double b, X_true, landmarkExternalDictionary[idx]);

            double bearing_noise = Deg2Rad * bearingNoiseFactor * normal.Sample(),
              range_noise = rangeNoiseFactor * normal.Sample();

            r += range_noise;
            b = ClampRad(b + bearing_noise);

            if (Math.Abs(b) <= bearing && r >= rangeMin && r <= rangeMax)
            {
                observations.Add(idx, Vector<double>.Build.DenseOfArray(new double[] { r, b }));
            }
        }

        return observations;
    }

    void CalculateMovement(out double d, out double a, Vector<double> currentState, Vector<double> targetPos)
    {
        d = (currentState.SubVector(0, 2) - targetPos).L2Norm();
        a = ClampRad(Math.Atan2(targetPos[1] - currentState[1], targetPos[0] - currentState[0]) - currentState[2]);
    }

    // Update both robot and landmarks
    void ObservationUpdate(ref Vector<double> X, ref Matrix<double> P, Vector<double> X_rob_actual, Dictionary<int, Vector<double>> observedLandmarks)
    {
        foreach (var landmark in observedLandmarks)
        {
            int world_id = landmark.Key;
            Vector<double> z_measure = landmark.Value;


            //Observation Update
            double r_measure = z_measure[0];
            double b_measure = z_measure[1];

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
                double x_r = X[0];
                double y_r = X[1];
                double theta_r = X[2];
                double alpha = ClampRad(theta_r + b_measure);

                Vector<double> X_lm_new = Vector<double>.Build.DenseOfArray(new double[]
                {x_r + r_measure*Math.Cos(alpha),
                    y_r + r_measure*Math.Sin(alpha) });

                Vector<double> X_new = Vector<double>.Build.DenseOfArray(new double[n_states + 2]);
                X_new.SetSubVector(0, n_states, X);
                X_new.SetSubVector(n_states, 2, X_lm_new);

                X = X_new; 

                Matrix<double> Jxr = Matrix<double>.Build.DenseOfArray(new double[2, 3]
                {{1, 0, -r_measure*Math.Sin(alpha)},
                {0, 1, r_measure*Math.Cos(alpha)}});

                Matrix<double> Jz = Matrix<double>.Build.DenseOfArray(new double[2, 2]
                {{Math.Cos(alpha), -r_measure*Math.Sin(alpha)},
                {Math.Sin(alpha), r_measure*Math.Cos(alpha)}});

                Matrix<double> Pvv = P.SubMatrix(0, 3, 0, 3);
                //Debug.Log("Pvv" + Pvv.ToString());

                Matrix<double> right_top = Pvv * Jxr.Transpose();
                Matrix<double> right_buttom = Jxr * Pvv * Jxr.Transpose() + Jz * R * Jz.Transpose();
                //Matrix<double> right_buttom = Matrix<double>.Build.DenseIdentity(2, 2) * 1000;

                P = P.DiagonalStack(right_buttom);

                P.SetSubMatrix(0, n_states, right_top);
                P.SetSubMatrix(n_states, 0, right_top.Transpose());

                if (n_states > 3)
                {
                    Matrix<double> Pmv = P.SubMatrix(3, n_states - 3, 0, 3);
                    Matrix<double> right_middle = Pmv * Jxr.Transpose();

                    P.SetSubMatrix(3, n_states, right_middle);
                    P.SetSubMatrix(n_states, 3, right_middle.Transpose());
                }

                P = 0.5f * (P + P.Transpose());
            }

            //Debug.Log("Old landmark re-observed: " + world_id);
            int local_id = worldToLocalDictionary[world_id];
            int index = 3 + local_id * 2;

            //Estimate measurement based on estimate robot states and estimate landmark states 
            Vector<double> z_est = ObervationEstimate(X, X.SubVector(index, 2));
            Matrix<double> h = LmJacobian(X, X.SubVector(index, 2));

            Matrix<double> H = Matrix<double>.Build.DenseOfArray(new double[2, X.Count]);
            H.SetSubMatrix(0, 0, h);
            H.SetSubMatrix(0, index, -h.SubMatrix(0, 2, 0, 2));

            Matrix<double> inov = H * P * H.Transpose() + R;

            //Kalman gain
            Matrix<double> K = P * H.Transpose() * inov.Inverse();
            Vector<double> diff = z_measure - z_est;
            diff[1] = ClampRad(diff[1]);

            Vector<double> X_inc = K * diff;

            X += X_inc;
            X[2] = ClampRad(X[2]);

            P = (Matrix<double>.Build.DenseIdentity(X.Count, X.Count) - K * H) * P;
        }
    }

    public void ResetAll()
    {
        robotLandmarkIndex = 0;
        landmarkExternalDictionary.Clear();
        worldToLocalDictionary.Clear();
        InitializeRobotState(0, 0, 0);
        targetDis = 0;
        targetAng = 0;
    }

    double ClampRad(double ang)
    {
        double new_ang = ang;

        if (ang > Math.PI)
            new_ang -= 2 * Math.PI;
        else if (ang < -Math.PI)
            new_ang += 2 * Math.PI;

        return new_ang;
    }
}
