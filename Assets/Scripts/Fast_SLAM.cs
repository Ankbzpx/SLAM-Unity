using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System;
using System.Linq;
using Random = System.Random;

[RequireComponent(typeof(Fast_Visualizer))]
public class Fast_SLAM : MonoBehaviour
{
    [SerializeField]
    int numOfParticles = 15;

    //States vector
    Vector<double> X_rob_actual;

    Vector<double>[] X_particles;

    double[] X_weight;

    //landmark mean covariance
    List<Vector<double>>[] X_m;
    List<Matrix<double>>[] P_m;

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

    Fast_Visualizer fastVisualizer;
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
        fastVisualizer = GetComponent<Fast_Visualizer>();
    }
    // Start is called before the first frame update
    void Start()
    {
        R = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { rangeNoiseFactor, Deg2Rad * bearingNoiseFactor });
        Q = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { Math.Pow(transNoiseFactor, 2), Math.Pow(transNoiseFactor, 2), Math.Pow(Deg2Rad * angNoiseFactor, 2) });

        X_particles = new Vector<double>[numOfParticles];
        X_weight = new double[numOfParticles];
        X_m = new List<Vector<double>>[numOfParticles];
        P_m = new List<Matrix<double>>[numOfParticles];

        for (int i = 0; i < numOfParticles; i++)
        {
            X_m[i] = new List<Vector<double>>();
            P_m[i] = new List<Matrix<double>>();
        }

        InitializeRobotState(0, 0, 0);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (absAng > 0)
        {
            double delta_ang = Time.fixedDeltaTime * rotVelocity;
            Fast_Update(0, Math.Sign(targetAng)* delta_ang);

            absAng -= delta_ang;
        }
        else
        {
            if (targetDis > 0)
            {
                double delta_dis = Time.fixedDeltaTime * transVelocity;
                Fast_Update(delta_dis, 0);
                targetDis -= delta_dis;
            }
        }
    }

    public void SetTargetWorldPoints(double world_x, double world_y)
    {
        CalculateTargetPose(X_rob_actual, world_x, world_y, out targetDis, out targetAng, out absAng);
    }

    void Fast_Update(double deltaDis, double deltaAng)
    {
        if (deltaDis == 0)
        {
            //Ground Truth with noise
            StatesUpdate(ref X_rob_actual, deltaDis + deltaDis * transNoiseFactor * normal.Sample(),
                deltaAng + deltaDis * Deg2Rad * angNoiseFactor * normal.Sample());

            //Paticles are sampled from the proposal distribution
            for (int i = 0; i < X_particles.Length; i++)
            {
                //Motion model
                StatesUpdate(ref X_particles[i], deltaDis + deltaDis * transNoiseFactor * normal.Sample(),
                deltaAng + deltaDis * Deg2Rad * angNoiseFactor * normal.Sample());
            }

            fastVisualizer.Visualize(X_particles, X_rob_actual);

            return;
        }

        //Ground Truth with noise
        StatesUpdate(ref X_rob_actual, deltaDis + deltaDis * transNoiseFactor * normal.Sample(), 
            deltaAng + deltaDis * Deg2Rad * angNoiseFactor * normal.Sample());
        
        //Observation from ground truth
        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);


        //Paticles are sampled from the proposal distribution
        for (int i = 0; i < X_particles.Length; i++)
        {
            //Motion model
            StatesUpdate(ref X_particles[i], deltaDis + deltaDis * transNoiseFactor * normal.Sample(),
            deltaAng + deltaDis * Deg2Rad * angNoiseFactor * normal.Sample());

            Matrix<double> r = RotationMatrix(X_particles[i]);

            //Initialize robot mean, cov, observation cov
            Matrix<double> s = r * Q * r.Transpose();
            Vector<double> m = X_particles[i];
            Matrix<double> Qt = R;

            Vector<double> X_old = m;
            Matrix<double> P_old = s;

            //Debug.Log("Particle " + i + " old: " + X_old.ToString());

            #region Observation_update
            //Observation update
            //Interatively refine proposal distribution with observation
            foreach (var m_obs in observedLandmarks)
            {
                int worldId = m_obs.Key;

                if (!worldToLocalDictionary.ContainsKey(worldId))
                    continue;

                Vector<double> z = m_obs.Value;
                Vector<double> z_pred = ObervationEstimate(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);

                Vector<double> z_diff = z - z_pred;
                z_diff[1] = ClampRad(z_diff[1]);

                //Debug.Log("Measure measurement:" +z_diff.L2Norm());

                Matrix<double> hx = Hx(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);   //2x3
                Matrix<double> hm = Hm(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);   //2x2

                Qt += hm * R * hm.Transpose(); //2x2


                s = (hx.Transpose() * Qt.Inverse() * hx + s.Inverse()).Inverse();     //3x3
                m += s * hx.Transpose() * Qt.Inverse() * z_diff;     //3x1            
            }

            //Sample from proposal distribution
            MatrixNormal mn = new MatrixNormal(m.ToColumnMatrix(), s, Matrix<double>.Build.DenseIdentity(1, 1));
            Matrix<double> sample = mn.Sample();

            Vector<double> X_new = Vector<double>.Build.DenseOfArray(new double[3] { sample[0, 0], sample[1, 0], sample[2, 0]});

            //Debug.Log("Mean " + i + " : " + m.ToString());
            //Debug.Log("X_actual " + i + " : " + X_rob_actual.ToString());

            //Update particle
            X_particles[i] = X_new;
            X_particles[i][2] = ClampRad(X_particles[i][2]);

            #endregion

            #region weight_update
            //Compute sample weight from new sampled particles
            //compute likelihood p(z|x)
            double likelihood = 1;
            foreach (var m_obs in observedLandmarks)
            {
                int worldId = m_obs.Key;

                if (!worldToLocalDictionary.ContainsKey(worldId))
                    continue;

                Vector<double> z = m_obs.Value;
                Vector<double> z_pred = ObervationEstimate(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);

                Matrix<double> hm = Hm(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);   //2x2
                Matrix<double> hx = Hx(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);   //2x3
                //Independent
                likelihood *= Gaussian_evaluation(z, z_pred, hm * R * hm.Transpose());
            }

            //Debug.Log(i + " likelihood: " + likelihood);


            double prior = Gaussian_evaluation(X_new, X_old, P_old);
            //Debug.Log(i + " prior: " + prior);

            double prop = Gaussian_evaluation(X_new, m , s);
            //Debug.Log(i + " prop: " + prop);

            //Update weight
            X_weight[i] *= likelihood* prior/ (prop + 1e-5);

            //Debug.Log("X weight: " + X_weight[i]);

            #endregion

            #region ekf_update

            //EKF update for each associated landmarks
            foreach (var m_obs in observedLandmarks)
            {
                int worldId = m_obs.Key;

                if (!worldToLocalDictionary.ContainsKey(worldId))
                    continue;

                Vector<double> z = m_obs.Value;
                Vector<double> z_pred = ObervationEstimate(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);

                Vector<double> z_diff = z - z_pred;
                z_diff[1] = ClampRad(z_diff[1]);

                Matrix<double> hm = Hm(X_particles[i], X_m[i][worldToLocalDictionary[worldId]]);   //2x2
                Matrix<double> inov = hm * P_m[i][worldToLocalDictionary[worldId]] * hm.Transpose() + R;    //2x2

                //Kalman gain
                Matrix<double> K = P_m[i][worldToLocalDictionary[worldId]] * hm.Transpose() * inov.Inverse();

                Vector<double> X_inc = K * z_diff;

                X_m[i][worldToLocalDictionary[worldId]] += X_inc;
                P_m[i][worldToLocalDictionary[worldId]] = (Matrix<double>.Build.DenseIdentity(2, 2) - K * hm) * P_m[i][worldToLocalDictionary[worldId]];
            }

            #endregion

        }

        //Normalize
        double count = 0;
        double[] cdf = new double[numOfParticles];
        double sum = X_weight.Sum();

        //Debug.Log("Sum: " + sum);

        for (int i = 0; i < X_particles.Length; i++)
        {
            X_weight[i] = X_weight[i] / sum;
            count += X_weight[i];
            cdf[i] = count;
        }

        fastVisualizer.Visualize(X_particles, X_rob_actual);

        //Resample
        Vector<double>[] new_X_particles = new Vector<double>[numOfParticles];
        List<Vector<double>>[] new_X_m = new List<Vector<double>>[numOfParticles];
        List<Matrix<double>>[] new_P_m = new List<Matrix<double>>[numOfParticles];

        for (int i = 0; i < X_particles.Length; i++)
        {
            int idx = GetCDFIndex(cdf);
            new_X_particles[i] = X_particles[idx].Clone();
            new_X_m[i] = new List<Vector<double>>(X_m[idx]);
            new_P_m[i] = new List<Matrix<double>>(P_m[idx]);
            X_weight[i] = 1 / (double)numOfParticles;
        }

        X_particles = new_X_particles;
        X_m = new_X_m;
        P_m = new_P_m;

        //Loop through all landmarks to check if new landmarks are observed
        foreach (var m_obs in observedLandmarks)
        {
            int world_id = m_obs.Key;

            //If it is a new landmark
            if (!worldToLocalDictionary.ContainsKey(world_id))
            {
                // Register the new landmark
                worldToLocalDictionary.Add(world_id, robotLandmarkIndex);
                robotLandmarkIndex++;

                for (int i = 0; i < X_particles.Length; i++)
                {
                    //Debug.Log(X_m[i].ToString());

                    X_m[i].Add(Vector<double>.Build.DenseOfArray(new double[]
                                    {X_particles[i][0] + m_obs.Value[0]*Math.Cos(X_particles[i][2] + m_obs.Value[1]),
                                     X_particles[i][1] + m_obs.Value[0]*Math.Sin(X_particles[i][2] + m_obs.Value[1])}));

                    Matrix<double> Jz = Matrix<double>.Build.DenseOfArray(new double[2, 2]
                                    {{Math.Cos(X_particles[i][2] + m_obs.Value[1]), -m_obs.Value[0]*Math.Sin(X_particles[i][2] + m_obs.Value[1])},
                                     {Math.Sin(X_particles[i][2] + m_obs.Value[1]),  m_obs.Value[0]*Math.Cos(X_particles[i][2] + m_obs.Value[1])}});

                    P_m[i].Add(Jz * R * Jz.Transpose());

                    fastVisualizer.VisualizeLandmarkRegisteration(i, X_particles[i][0] + m_obs.Value[0] * Math.Cos(X_particles[i][2] + m_obs.Value[1]), 
                        X_particles[i][1] + m_obs.Value[0] * Math.Sin(X_particles[i][2] + m_obs.Value[1]));
                }
            }
        }

        fastVisualizer.Visualize(X_particles, X_rob_actual);
        fastVisualizer.VisualizeMeasurement(observedLandmarks);
        fastVisualizer.VisualizeLandmarkParticles(X_m);
    }

    int GetCDFIndex(double[] cdf)
    {
        double s = random.NextDouble();
        int index = 0;

        for (int i = 0; i < numOfParticles; i++)
        {
            if (s <= cdf[i])
            {
                index = i;
                break;
            }
        }
        return index;
    }

    double Gaussian_evaluation(Vector<double> target, Vector<double> m, Matrix<double> s)
    {
        MatrixNormal mn = new MatrixNormal(m.ToColumnMatrix(), s, Matrix<double>.Build.DenseIdentity(1, 1));

        return mn.Density(target.ToColumnMatrix());
    }


    void InitializeRobotState(double x, double y, double theta)
    {
        X_rob_actual = Vector<double>.Build.DenseOfArray(new double[] { x, y, theta });     

        for (int i = 0; i < X_particles.Length; i++)
        {
            X_particles[i] = Vector<double>.Build.DenseOfArray(new double[] { x, y, theta });
            X_weight[i] = 1 / (double)X_particles.Length;
        }

        fastVisualizer.VisualizerRobotRegistration(numOfParticles, (float)x, (float)y, (float)theta);
    }

    public void InitializeLandmarks(int numOfLandmarks, double worldLimitX, double worldLimitY)
    {
        landmarkExternalDictionary.Clear();

        landmarkExternalDictionary.Add(0, Vector<double>.Build.DenseOfArray(new double[] { 0.5, 0.5}));
        fastVisualizer.VisualizeTrueLandmarks(0, 0.5f, 0.5f);

        for (int i = 1; i < numOfLandmarks; i++)
        {
            double x = random.NextDouble() * worldLimitX;
            double y = random.NextDouble() * worldLimitY;

            landmarkExternalDictionary.Add(i, Vector<double>.Build.DenseOfArray(new double[] { x, y }));
            fastVisualizer.VisualizeTrueLandmarks(i, (float)x, (float)y);
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

    Vector<double> ObervationEstimate(Vector<double> X_est, Vector<double> X_lm_est)
    {
        CalculateMovement(out double r, out double b, X_est, X_lm_est);
        return Vector<double>.Build.DenseOfArray(new double[] { r, b });
    }

    Matrix<double> F(double dis, double ang)
    {
        return Matrix<double>.Build.DenseOfArray(new double[3, 3] { 
            { 1, 0, -dis * Math.Sin(ang) }, 
            { 0, 1, dis * Math.Cos(ang) }, 
            { 0, 0, 1 } }); ;
    }

    Matrix<double> RotationMatrix(Vector<double> X_est)
    {
        return Matrix<double>.Build.DenseOfArray(new double[3, 3] { 
            { Math.Cos(X_est[2]), -Math.Sin(X_est[2]), 0 }, 
            { Math.Sin(X_est[2]), Math.Cos(X_est[2]), 0 }, 
            { 0, 0, 1 } });
    }


    Matrix<double> Hx(Vector<double> X_est, Vector<double> X_lm_est)
    {
        Vector<double> d = X_est.SubVector(0, 2) - X_lm_est;
        double r = d.L2Norm();

        return Matrix<double>.Build.DenseOfArray(new double[2, 3]
            {{(d[0])/r, (d[1])/r, 0},
            {(-d[1])/Math.Pow(r, 2), (d[0])/Math.Pow(r, 2), -1 }});
    }

    Matrix<double> Hm(Vector<double> X_est, Vector<double> X_lm_est)
    {
        return -Hx(X_est, X_lm_est).SubMatrix(0, 2, 0, 2);
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
