using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Differentiation;
using System.Collections.Generic;
using UnityEngine;
using System;
using Random = System.Random;

public class GraphSLAM : MonoBehaviour
{
    [SerializeField]
    int recordInterval = 5;

    int count = 0;

    [SerializeField]
    int optimizeEpoch = 20;

    public int historyLength = 100;

    bool allowOptimization = false, isOptimizing = false;

    List<Vector<double>> robotTrace_predict = new List<Vector<double>>();
    List<Vector<double>> robotTrace_actual = new List<Vector<double>>();
    List<Dictionary<int, Vector<double>>> landmarkTrace = new List<Dictionary<int, Vector<double>>>();

    List<List<int>> graph = new List<List<int>>();

    //States vector
    Vector<double> X_rob_predict, X_rob_actual;

    public double transNoiseFactor = 0.1, angNoiseFactor = 5;
    public double rangeNoiseFactor = 0.01, bearingNoiseFactor = 1;

    Normal normal = Normal.WithMeanPrecision(0, 1);
    Matrix<double> R, Q;

    Dictionary<int, Vector<double>> observedLandmarks = new Dictionary<int, Vector<double>>();

    double targetDis = 0, targetAng = 0, absAng = 0;
    //bool isTargetReached = false;

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

    // Dictionary that stores associated landmarks
    public Dictionary<int, Vector<double>> landmarkDictionary =
        new Dictionary<int, Vector<double>>();

    GraphVisualizer graphVisualizer;

    const double Deg2Rad = Math.PI / 180;
    const double Rad2Deg = 180 / Math.PI;
    Random random;

    private void Awake()
    {
        random = new Random();
        graphVisualizer = GetComponent<GraphVisualizer>();
    }
    // Start is called before the first frame update
    void Start()
    {
        R = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { rangeNoiseFactor, Deg2Rad * bearingNoiseFactor });
        Q = Matrix<double>.Build.DenseOfDiagonalArray(new double[] { Math.Pow(transNoiseFactor, 2), Math.Pow(transNoiseFactor, 2), Math.Pow(Deg2Rad * angNoiseFactor, 2) });
        InitializeRobotState(0f, 0f, 0f);
    }

    void FixedUpdate()
    {
        if (isOptimizing)
            return;

        if (absAng > 0)
        {
            double delta_ang = Time.fixedDeltaTime * rotVelocity;
            Graph_Update(0, Math.Sign(targetAng) * delta_ang);
            absAng -= delta_ang;
        }
        else
        {
            if (targetDis > 0)
            {
                double delta_dis = Time.fixedDeltaTime * transVelocity;
                Graph_Update(delta_dis, 0f);
                targetDis -= delta_dis;
            }
        }
    }

    public void SetTargetWorldPoints(double world_x, double world_y)
    {
        CalculateTargetPose(X_rob_actual, world_x, world_y, out targetDis, out targetAng, out absAng);
    }

    void Graph_Update(double deltaDis, double deltaAng)
    {
        allowOptimization = false;

        //Shall not happen
        if (robotTrace_predict.Count != robotTrace_actual.Count)
            return;

        //Debug.Log("Graph update");

        ////Avoid Overflow
        //if (robotTrace_predict.Count >= historyLength)
        //{
        //    //Needs Debugging!
        //    robotTrace_predict.RemoveAt(0);
        //    robotTrace_actual.RemoveAt(0);
        //}

        double ang_noise = deltaDis / 1 * Deg2Rad * angNoiseFactor * normal.Sample(),
            dis_noise = deltaDis / 1 * transNoiseFactor * normal.Sample();

        StatesUpdate(ref X_rob_actual, deltaDis + dis_noise, deltaAng + ang_noise);
        StatesUpdate(ref X_rob_predict, deltaDis, deltaAng);
        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);

        graphVisualizer.Visualize(X_rob_predict, X_rob_actual, false);

        if (deltaDis != 0)
        {
            if (count == recordInterval)
            {
                graphVisualizer.Visualize(X_rob_predict, X_rob_actual, true);

                BuildGraph(observedLandmarks, landmarkTrace, ref graph);

                landmarkTrace.Add(observedLandmarks);
                robotTrace_predict.Add(X_rob_predict.Clone());
                robotTrace_actual.Add(X_rob_actual.Clone());

                List<List<int>> cycles = new List<List<int>>();
                //cycles = FindCycle(graph, landmarkTrace.Count - 1, landmarkTrace.Count - 1, new List<int>(), new List<int>(), cycles);
                cycles.Add(FindCycleSimplified(graph));

                if (cycles.Count != 0 && cycles[0].Count != 0)
                {
                    //Debug.Log("Debug graph:");
                    //for (int h = 0; h < graph.Count; h++)
                    //{
                    //    string str = h + "-";
                    //    for (int i = 0; i < graph[h].Count; i++)
                    //    {
                    //        str += graph[h][i] + "-";
                    //    }

                    //    Debug.Log("Entry: " + str);
                    //}

                    //Debug.Log("Print all cycle: ");
                    //for (int h = 0; h < cycles.Count; h++)
                    //{
                    //    string str = "";

                    //    for (int i = 0; i < cycles[h].Count; i++)
                    //    {
                    //        str += cycles[h][i] + "-";
                    //    }

                    //    Debug.Log("Cycle: " + str);
                    //}


                    allowOptimization = true;
                    Optimize(cycles);
                }

                count = 0;
            }
               
            count++;
        }
    }

    void BuildGraph(Dictionary<int, Vector<double>> newObservations, List<Dictionary<int, Vector<double>>> landmarkTrace, 
        ref List<List<int>> graph)
    {
        //Add empty entry for new observation
        graph.Add(new List<int>());

        //foreach pose corresponded observations
        for (int i = 0; i < landmarkTrace.Count; i++)
        {
            //For each observed landmark at given pose
            foreach (int lmidx in landmarkTrace[i].Keys)
            {
                //If the new observation has the same landmark as the old ones
                if (newObservations.ContainsKey(lmidx))
                {
                    graph[i].Add(landmarkTrace.Count);
                    graph[landmarkTrace.Count].Add(i);

                    //Only care if two poses share same observation
                    break;
                }
            }
        }

        if (landmarkTrace.Count != 0)
        {
            //Pose of t-1 and t are connected
            if (!graph[landmarkTrace.Count].Contains(landmarkTrace.Count - 1))
                graph[landmarkTrace.Count].Add(landmarkTrace.Count - 1);

            if (!graph[landmarkTrace.Count - 1].Contains(landmarkTrace.Count))
                graph[landmarkTrace.Count - 1].Add(landmarkTrace.Count);
        }
    }

    List<int> FindCycleSimplified(List<List<int>> graph)
    {
        List<int> cycle = new List<int>();
        int index = graph.Count - 1;

        for (int i = 0; i < graph.Count - 2; i++)
        {
            if (graph[graph.Count - 1].Contains(i))
            {
                index = i;
                break;
            }
        }

        if (index != graph.Count - 1)
        {
            cycle.Add(graph.Count - 1);

            for (int i = index; i < graph.Count - 1; i++)
            {
                cycle.Add(i);
            }
        }

        return cycle;
    }

    //Depth first search for finding cycle
    List<List<int>> FindCycle(List<List<int>> graph, int start, int current, List<int> visited, List<int> cycle, List<List<int>> cycles)
    {
        if (!visited.Contains(current))
        {
            visited.Add(current);
            cycle.Add(current);
            foreach (int v in graph[current])
            {
                if (!visited.Contains(v))
                {
                    if (graph[v].Count == 1)
                    {
                        visited.Add(v);
                    }
                    else
                    {
                        if (start != current && graph[v].Contains(start))
                        {
                            visited.Add(v);
                            cycle.Add(v);
                            cycles.Add(new List<int>(cycle));
                            cycle.Remove(cycle.Count - 1);
                        }
                        else
                        {
                            FindCycle(graph, start, v, visited, cycle, cycles);
                        }
                    }
                }
            }

            cycle.Remove(cycle.Count - 1);
        }

        return cycles;
    }


    void Optimize(List<List<int>> cycles)
    {
        if (!allowOptimization)
            return;

        isOptimizing = true;

        for (int i = 0; i < optimizeEpoch; i++)
        {
            OptimizeOnce(ref robotTrace_predict, landmarkTrace, cycles);
        }

        //Debug.Log("Before update: " + X_rob_predict.ToString());
        X_rob_predict = robotTrace_predict[robotTrace_predict.Count - 1];

        //Debug.Log("After update: " + robotTrace_predict[robotTrace_predict.Count - 1].ToString());
        //Debug.Log("Actual pose: " + X_rob_actual.ToString());

        isOptimizing = !graphVisualizer.OptimizedVisualization(robotTrace_predict);
    }

    void OptimizeOnce(ref List<Vector<double>> robotTrace_predict, List<Dictionary<int, Vector<double>>> landmarkTrace, List<List<int>> cycles)
    {
        bool isConnected = true;
        List<int> poseList = new List<int>();
        List<int[]> indexPairList = new List<int[]>();
        List<int[]> landmarkPairList = new List<int[]>();

        GetPoseListAndPairs(cycles, ref poseList, ref indexPairList, ref landmarkPairList);

        double error = 100f;
        int optEpoch = optimizeEpoch;

        while (error > 1f && optEpoch > 0)
        {
            error = 0f;

            List<Vector<double>> robotTraceWithConstraints = new List<Vector<double>>();
            foreach (int idx in poseList)
            {
                robotTraceWithConstraints.Add(robotTrace_predict[idx].SubVector(0, 2));
            }

            //Return if no constraints exist in current trajectory
            if (robotTraceWithConstraints.Count == 0)
                return;

            //Start optimization
            Vector<double> robotTrace_optimize_vector = ListToVector(robotTraceWithConstraints);
            Vector<double> b = Vector<double>.Build.SparseOfArray(new double[robotTrace_optimize_vector.Count]);
            Matrix<double> H = Matrix<double>.Build.SparseOfArray(new double[robotTrace_optimize_vector.Count, robotTrace_optimize_vector.Count]);


            int connectedCount = 0;

            for (int i = 0; i < indexPairList.Count; i++)
            {
                int[] indexPair = indexPairList[i];
                int[] landmarkPair = landmarkPairList[i];

                //Observation dictionary of i and j
                Dictionary<int, Vector<double>> lmObs_i = landmarkTrace[landmarkPair[0]];
                Dictionary<int, Vector<double>> lmObs_j = landmarkTrace[landmarkPair[1]];

                Vector<double> X_i = robotTrace_predict[landmarkPair[0]]; /*robotTrace_optimize_vector.SubVector(2 * indexPair[0], 2);*/
                Vector<double> X_j = robotTrace_predict[landmarkPair[1]]; /*robotTrace_optimize_vector.SubVector(2 * indexPair[1], 2);*/

                Matrix<double> H_ii = H.SubMatrix(2 * indexPair[0], 2, 2 * indexPair[0], 2);
                Matrix<double> H_ij = H.SubMatrix(2 * indexPair[0], 2, 2 * indexPair[1], 2);
                Matrix<double> H_ji = H.SubMatrix(2 * indexPair[1], 2, 2 * indexPair[0], 2);
                Matrix<double> H_jj = H.SubMatrix(2 * indexPair[1], 2, 2 * indexPair[1], 2);

                Vector<double> b_i = b.SubVector(2 * indexPair[0], 2);
                Vector<double> b_j = b.SubVector(2 * indexPair[1], 2);

                connectedCount = 0;

                //Motion constraints (adjacent nodes)
                if (Math.Abs(landmarkPair[0] - landmarkPair[1]) == 1)
                {
                    Vector<double> err = GetErrorMotion(X_i, X_j, robotTrace_actual[landmarkPair[0]] - robotTrace_actual[landmarkPair[1]]);
                    Matrix<double> A = GetJacobianA();
                    Matrix<double> B = -GetJacobianA();
                    Matrix<double> omega = GetOmegaMotion(X_i, X_j);

                    //Debug.Log("Motion error: " + err.ToString());
                    error += err.L2Norm();

                    //Omega, update H, b
                    H_ii += A.Transpose() * omega * A;
                    H_ij += A.Transpose() * omega * B;
                    H_ji += B.Transpose() * omega * A;
                    H_jj += B.Transpose() * omega * B;

                    b_i += A.Transpose() * omega * err;
                    b_j += B.Transpose() * omega * err;

                    connectedCount++;

                }

          
                //Observation constraint
                //Loop through all landmarks observed at i
                foreach (int lmIdx in lmObs_i.Keys)
                {
                    //If same landmark is also observed at j
                    if (lmObs_j.ContainsKey(lmIdx))
                    {
                        Vector<double> err = GetErrorObservation(X_i, X_j, lmObs_i[lmIdx], lmObs_j[lmIdx]);
                        //Debug.Log("Error x: " + err.L2Norm());

                        if (!double.IsNaN(err.L2Norm()))
                        {
                            error += err.L2Norm();
                            //Debug.Log("Error vector: " + err.ToString());

                            //Matrix<double> A = CalculateJacobianA(X_i, X_j, lmObs_i[lmIdx], lmObs_j[lmIdx]);
                            //Matrix<double> B = CalculateJacobianB(X_i, X_j, lmObs_i[lmIdx], lmObs_j[lmIdx]);

                            Matrix<double> A = GetJacobianA(X_i, lmObs_i[lmIdx]);
                            Matrix<double> B = -GetJacobianA(X_j, lmObs_j[lmIdx]);
                            Matrix<double> omega = GetOmegaObservation(X_i, X_j, lmObs_i[lmIdx], lmObs_j[lmIdx]);


                            //Omega, update H, b
                            H_ii += A.Transpose() * omega * A;
                            H_ij += A.Transpose() * omega * B;
                            H_ji += B.Transpose() * omega * A;
                            H_jj += B.Transpose() * omega * B;

                            b_i += A.Transpose() * omega * err;
                            b_j += B.Transpose() * omega * err;

                            connectedCount++;
                        }
                        else
                        {
                            //Debug.Log("Nan detected.");
                        }                     
                    }
                }

                H.SetSubMatrix(2 * indexPair[0], 2, 2 * indexPair[0], 2, H_ii);
                H.SetSubMatrix(2 * indexPair[0], 2, 2 * indexPair[1], 2, H_ij);
                H.SetSubMatrix(2 * indexPair[1], 2, 2 * indexPair[0], 2, H_ji);
                H.SetSubMatrix(2 * indexPair[1], 2, 2 * indexPair[1], 2, H_jj);

                b.SetSubVector(2 * indexPair[0], 2, b_i);
                b.SetSubVector(2 * indexPair[1], 2, b_j);

                if (connectedCount == 0)
                {
                    string lm = "";
                    foreach (var idx in landmarkPair)
                    {
                        lm += idx + ", ";
                    }

                    Debug.Log("landmark Pair not connected: " + lm);

                    //string id = "";
                    //foreach (var idx in indexPair)
                    //{
                    //    id += idx + ", ";
                    //}

                    //Debug.Log("Index Pair not connected: " + id);

                    isConnected = false;
                }
            }

            if (isConnected)
            {
                //Debug.Log("Graph connected");
                //Debug.Log("Solving the optimization problem...");

                Matrix<double> H_11 = H.SubMatrix(0, 2, 0, 2);
                H_11 += Matrix<double>.Build.SparseIdentity(2);
                H.SetSubMatrix(0, 2, 0, 2, H_11);

                H = 0.5f * (H + H.Transpose());

                Vector<double> X_delta = H.Cholesky().Solve(-b);
                //Vector<double> X_delta = H.Solve(-b);

                //Debug.Log(X_delta.ToString());

                for (int i = 0; i < robotTrace_optimize_vector.Count; i++)
                {
                    if (!double.IsNaN(X_delta[i]) && !double.IsInfinity(X_delta[i]))
                    {
                        robotTrace_optimize_vector[i] += X_delta[i];
                    }
                }

                robotTraceWithConstraints = VectorToList(robotTrace_optimize_vector);

                for (int i = 0; i < poseList.Count; i++)
                {
                    robotTrace_predict[poseList[i]][0] = robotTraceWithConstraints[i][0];
                    robotTrace_predict[poseList[i]][1] = robotTraceWithConstraints[i][1];
                }
            }

            optEpoch--;
            Debug.Log("Error :" + error);
        }
    }

    Matrix<double> GetOmegaMotion(Vector<double> X_i, Vector<double> X_j)
    {
        Matrix<double> rt_i = GetRotationMatrix(X_i[2]);
        Matrix<double> rt_j = GetRotationMatrix(X_j[2]);

        return (rt_i * Q * rt_i.Transpose() + rt_j * Q * rt_j.Transpose()).Inverse().SubMatrix(0, 2, 0, 2);
    }

    Matrix<double> GetOmegaObservation(Vector<double> X_i, Vector<double> X_j, Vector<double> lm_i, Vector<double> lm_j)
    {
        //Matrix<double> rt_i = GetRotationMatrix(X_i[2]);
        //Matrix<double> rt_j = GetRotationMatrix(X_j[2]);

        //return (rt_i * Q * rt_i.Transpose() + rt_j * Q * rt_j.Transpose()).Inverse().SubMatrix(0, 2, 0, 2);

        Matrix<double> J_i = LmJacobian(X_i, lm_i);
        Matrix<double> J_j = LmJacobian(X_j, lm_j);

        return (J_i.Transpose() * R * J_i + J_j.Transpose() * R * J_j).Inverse().SubMatrix(0, 2, 0, 2);
    }

    Matrix<double> GetRotationMatrix(double theta)
    {
        return Matrix<double>.Build.DenseOfArray(new double[3, 3]
            {{Math.Cos(theta), -Math.Sin(theta), 0 }, 
            {Math.Sin(theta), Math.Cos(theta), 0}, 
            {0, 0, 1.0f }});
    }

    Matrix<double> LmJacobian(Vector<double> X_est, Vector<double> X_lm_est)
    {
        Vector<double> d = X_est.SubVector(0, 2) - X_lm_est;
        double r = (double)d.L2Norm();

        return Matrix<double>.Build.DenseOfArray(new double[2, 3]
            {{(d[0])/r, (d[1])/r, 0},
            {(-d[1])/Math.Pow(r, 2), (d[0])/Math.Pow(r, 2), -1 }});
    }

    Vector<double> GetErrorMotion(Vector<double> Xt, Vector<double> Xt_1, Vector<double> diff)
    {
        Vector<double> xt_obs = Xt_1 + diff;
        return Vector<double>.Build.DenseOfArray(new double[2] { xt_obs[0] - Xt[0], xt_obs[1] - Xt[1] });
    }

    Vector<double> GetErrorObservation(Vector<double> X_i, Vector<double> X_j, Vector<double> z_i, Vector<double> z_j)
    {
        double lm_i_x = X_i[0] + z_i[0] * Math.Cos(X_i[2] + z_i[1]);
        double lm_i_y = X_i[1] + z_i[0] * Math.Sin(X_i[2] + z_i[1]);

        double lm_j_x = X_j[0] + z_j[0] * Math.Cos(X_j[2] + z_j[1]);
        double lm_j_y = X_j[1] + z_j[0] * Math.Sin(X_j[2] + z_j[1]);

        return Vector<double>.Build.DenseOfArray(new double[2] { lm_j_x - lm_i_x, lm_j_y - lm_i_y });
    }

    Matrix<double> GetJacobianA()
    {
        return Matrix<double>.Build.DenseOfArray(new double[2, 2]
            {{-1, 0},
             {0, -1}});
    }

    Matrix<double> GetJacobianA(Vector<double> X, Vector<double> z)
    {
        return Matrix<double>.Build.DenseOfArray(new double[2, 2]
            {{-1, 0},
             {0, -1}});
    }

    void InitializeRobotState(double x, double y, double theta)
    {
        X_rob_predict = Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0 });
        X_rob_actual = X_rob_predict.Clone();

        observedLandmarks = ObserveLandmarks(X_rob_actual, rangeNoiseFactor, bearingNoiseFactor);

        BuildGraph(observedLandmarks, landmarkTrace, ref graph);
        robotTrace_predict.Add(X_rob_predict.Clone());
        robotTrace_actual.Add(X_rob_actual.Clone());
        landmarkTrace.Add(observedLandmarks);
        graphVisualizer.VisualizerRobotRegistration((float)x, (float)y, (float)theta);
    }

    public void InitializeLandmarks(int numOfLandmarks, double worldLimitX, double worldLimitY)
    {
        landmarkDictionary.Clear();

        //landmarkDictionary.Add(0, Vector<double>.Build.DenseOfArray(new double[] { 0.5f, 0.5f }));
        //graphVisualizer.VisualizerLandmarkRegistration(0, 0.5f, 0.5f);

        for (int i = 0; i < numOfLandmarks; i++)
        {
            double x = random.NextDouble() * worldLimitX;
            double y = random.NextDouble() * worldLimitY;

            landmarkDictionary.Add(i, Vector<double>.Build.DenseOfArray(new double[] { x, y }));
            graphVisualizer.VisualizerLandmarkRegistration(i, (float)x, (float)y);
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

    Dictionary<int, Vector<double>> ObserveLandmarks(Vector<double> X_true, double rangeNoiseFactor, double bearingNoiseFactor)
    {
        Dictionary<int, Vector<double>> observations = new Dictionary<int, Vector<double>>();

        foreach (int idx in landmarkDictionary.Keys)
        {
            CalculateMovement(out double r, out double b, X_true, landmarkDictionary[idx]);

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
        d = (double)(currentState.SubVector(0, 2) - targetPos).L2Norm();
        a = ClampRad(Math.Atan2(targetPos[1] - currentState[1], targetPos[0] - currentState[0]) - currentState[2]);
    }

    void GetPoseListAndPairs(List<List<int>> cycles, ref List<int> poseList, ref List<int[]> indexPairList, ref List<int[]> landmarkPairList)
    {
        //The latest pose
        int endPoint = cycles[0][0];

        foreach (List<int> cycle in cycles)
        {
            for (int i = 1; i < cycle.Count; i++)
            {
                if (!poseList.Contains(cycle[i]))
                {
                    poseList.Add(cycle[i]);
                }
                    
            }
        }
        poseList.Add(endPoint);


        foreach (List<int> cycle in cycles)
        {
            for (int i = 0; i < cycle.Count - 1; i++)
            {
                int[] lmPair = new int[2] { cycle[i], cycle[i + 1] };
                int[] idxPair = new int[2] { poseList.IndexOf(cycle[i]), poseList.IndexOf(cycle[i + 1]) };

                graphVisualizer.VisualizeConstraints(lmPair);

                if (!landmarkPairList.Contains(lmPair))
                    landmarkPairList.Add(lmPair);

                if (!indexPairList.Contains(idxPair))
                    indexPairList.Add(idxPair);
            }

            int[] lastLmPair = new int[2] { cycle[0], cycle[cycle.Count - 1]};
            int[] lastIdxPair = new int[2] { poseList.IndexOf(cycle[0]), poseList.IndexOf(cycle[cycle.Count - 1])};


            if (!landmarkPairList.Contains(lastLmPair))
                landmarkPairList.Add(lastLmPair);

            if (!indexPairList.Contains(lastIdxPair))
                indexPairList.Add(lastIdxPair);
        }
    }


    #region Utility
    double ClampRad(double ang)
    {
        double new_ang = ang;

        if (ang > Math.PI)
            new_ang -= 2 * Math.PI;
        else if (ang < -Math.PI)
            new_ang += 2 * Math.PI;

        return new_ang;
    }



    Vector<double> ListToVector(List<Vector<double>> robotTraceWithConstraints)
    {
        Vector<double> robStateVector = Vector<double>.Build.DenseOfArray(new double[robotTraceWithConstraints.Count * 2]);

        for (int i = 0; i < robotTraceWithConstraints.Count; i++)
        {
            robStateVector.SetSubVector(2 * i, 2, robotTraceWithConstraints[i]);
        }

        return robStateVector;
    }

    List<Vector<double>> VectorToList(Vector<double> robStateVector)
    {
        List<Vector<double>> robotTrace_actual = new List<Vector<double>>();

        for (int i = 0; i < robStateVector.Count / 2; i++)
        {
            robotTrace_actual.Add(robStateVector.SubVector(2 * i, 2));
        }

        return robotTrace_actual;
    }

    #endregion
}