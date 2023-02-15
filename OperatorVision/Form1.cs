using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;

namespace OperatorVision
{
    public partial class Form1 : Form
    {

        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint cButtons, uint dwExtraInfo);

        [DllImport("user32.dll")]
        static extern bool ClientToScreen(IntPtr hWnd, ref Point lpPoint);

        [DllImport("user32.dll")]
        public static extern long SetCursorPos(int x, int y);
        [DllImport("user32.dll", SetLastError = false)]
        static extern IntPtr GetDesktopWindow();
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool GetCursorPos(out Point point);

        [DllImport("User32.dll")]
        static extern int SetForegroundWindow(IntPtr point);

        //Mouse actions
        private const int MOUSEEVENTF_LEFTDOWN = 0x02;
        private const int MOUSEEVENTF_LEFTUP = 0x04;
        private const int MOUSEEVENTF_RIGHTDOWN = 0x08;
        private const int MOUSEEVENTF_RIGHTUP = 0x10;
        public Form1()
        {
            InitializeComponent();
        }
        struct neural_network_2hid
        {
            public int input_nodes;
            public int hidden_nodes_1;
            public int hidden_nodes_2;
            public int output_nodes;
            public double learning_grate;
            public string path;
            public Matrix<double> wih1;
            public Matrix<double> wh1h2;
            public Matrix<double> wh2o;
            public neural_network_2hid(int input_nodes, int hidden_nodes_1, int hidden_nodes_2, int output_nodes,
            double learning_grate, string path)
            {
                this.input_nodes = input_nodes;
                this.hidden_nodes_1 = hidden_nodes_1;
                this.hidden_nodes_2 = hidden_nodes_2;
                this.output_nodes = output_nodes;
                this.learning_grate = learning_grate;
                this.path = path;
                // матрицы весовых коэффициентов связей wih1 (weight input-hidden_1) между входным и скрытым 
                // wh1h2 
                // wh2o (weight hidden2-output) между скрытым и выходным
                if (path == null)
                {
                    double[,] array1 = new double[hidden_nodes_1, input_nodes];
                    for (int i = 0; i < hidden_nodes_1; i++)
                        for (int j = 0; j < input_nodes; j++)
                        {
                            double value = 0;
                            double limit = Math.Pow(hidden_nodes_1, -0.5);
                            while (value == 0 || value > limit || value < -limit)
                            {
                                Random rnd = new Random();
                                value = rnd.NextDouble() - 0.5;
                            }
                            array1[i, j] = value;
                        }
                    //
                    this.wih1 = DenseMatrix.OfArray(array1);
                    //
                    double[,] array3 = new double[hidden_nodes_2, hidden_nodes_1];
                    for (int i = 0; i < hidden_nodes_2; i++)
                        for (int j = 0; j < hidden_nodes_1; j++)
                        {
                            double value = 0;
                            double limit = Math.Pow(hidden_nodes_2, -0.5);
                            while (value == 0 || value > limit || value < -limit)
                            {
                                Random rnd = new Random();
                                value = rnd.NextDouble() - 0.5;
                            }
                            array3[i, j] = value;
                        }
                    //
                    this.wh1h2 = DenseMatrix.OfArray(array3);
                    //
                    double[,] array2 = new double[output_nodes, hidden_nodes_2];
                    for (int i = 0; i < output_nodes; i++)
                        for (int j = 0; j < hidden_nodes_2; j++)
                        {
                            double value = 0;
                            double limit = Math.Pow(output_nodes, -0.5);
                            while (value == 0 || value > limit || value < -limit)
                            {
                                Random rnd = new Random();
                                value = rnd.NextDouble() - 0.5;
                            }
                            array2[i, j] = value;
                        }
                    //
                    this.wh2o = DenseMatrix.OfArray(array2);
                }
                else // (mult, 200, 75, 4)
                {

                    List<string> all_lines = new List<string>();
                    using (StreamReader sr1 = new StreamReader(path))
                    {
                        string line;
                        while ((line = sr1.ReadLine()) != null)
                            all_lines.Add(line);
                    }
                    int a = 0;
                    double[,] array1 = new double[hidden_nodes_1, input_nodes];
                    for (int i = 0; i < hidden_nodes_1; i++)
                        for (int j = 0; j < input_nodes; j++, a++)
                            array1[i, j] = Convert.ToDouble(all_lines[a]);
                    a++;
                    this.wih1 = DenseMatrix.OfArray(array1);
                    //
                    double[,] array3 = new double[hidden_nodes_2, hidden_nodes_1];
                    for (int i = 0; i < hidden_nodes_2; i++)
                        for (int j = 0; j < hidden_nodes_1; j++, a++)
                            array3[i, j] = Convert.ToDouble(all_lines[a]);
                    a++;
                    this.wh1h2 = DenseMatrix.OfArray(array3);
                    //
                    double[,] array2 = new double[output_nodes, hidden_nodes_2];
                    for (int i = 0; i < output_nodes; i++)
                        for (int j = 0; j < hidden_nodes_2; j++, a++)
                            array2[i, j] = Convert.ToDouble(all_lines[a]);
                    //
                    this.wh2o = DenseMatrix.OfArray(array2);
                }

            }

            public double[,] train(double[,] inputs_d, double[,] targets_d) // проверить все матрицы на соответствие
            {
                Matrix<double> inputs = DenseMatrix.OfArray(inputs_d);
                Matrix<double> targets = DenseMatrix.OfArray(targets_d);
                Matrix<double> hidden_input1 = wih1 * inputs;
                for (int i = 0; i < hidden_input1.RowCount; i++)
                    for (int j = 0; j < hidden_input1.ColumnCount; j++)
                        hidden_input1[i, j] = 1 / (1 + Math.Exp(-hidden_input1[i, j])); // V1 = 1 / (1 + exp^(V2)) // hidden_input -> hidden_output //sigma activation
                Matrix<double> hidden_input2 = wh1h2 * hidden_input1;
                for (int i = 0; i < hidden_input2.RowCount; i++)
                    for (int j = 0; j < hidden_input2.ColumnCount; j++)
                        hidden_input2[i, j] = 1 / (1 + Math.Exp(-hidden_input2[i, j])); // V1 = 1 / (1 + exp^(V2)) // hidden_input -> hidden_output //sigma activation
                                                                                        ///////
                                                                                        //расчитать входящие сигналы для выходного слоя
                                                                                        //расчитать исходящие сигналы выходного слоя 
                Matrix<double> final_input = wh2o * hidden_input2;
                for (int i = 0; i < final_input.RowCount; i++)
                    for (int j = 0; j < final_input.ColumnCount; j++)
                        final_input[i, j] = 1 / (1 + Math.Exp(-final_input[i, j])); // V1 = 1 / (1 + exp^(V2)) // final_input -> final_output //sigma activation   
                                                                                    // ошибка выходного слоя =(целевое значение - фактическое)
                Matrix<double> output_errors = targets - final_input;
                Matrix<double> hidden_errors2 = wh2o.Transpose() * output_errors;
                Matrix<double> hidden_errors1 = wh1h2.Transpose() * hidden_errors2;
                // обновить весовые коэффициенты между скрытым и выходным
                // промежуточная матрица ???? проверить на соответсвие
                double[,] array1 = new double[final_input.RowCount, final_input.ColumnCount];
                for (int i = 0; i < final_input.RowCount; i++)
                    for (int j = 0; j < final_input.ColumnCount; j++)
                        array1[i, j] = output_errors[i, j] * final_input[i, j] * (1 - final_input[i, j]);
                Matrix<double> M1 = DenseMatrix.OfArray(array1);
                //
                wh2o += learning_grate * M1 * hidden_input2.Transpose();
                // обновить весовые коэффициенты между входным и скрытым
                double[,] array2 = new double[hidden_input2.RowCount, hidden_input2.ColumnCount];
                for (int i = 0; i < hidden_input2.RowCount; i++)
                    for (int j = 0; j < hidden_input2.ColumnCount; j++)
                        array2[i, j] = hidden_errors2[i, j] * hidden_input2[i, j] * (1 - hidden_input2[i, j]);
                Matrix<double> M2 = DenseMatrix.OfArray(array2);
                wh1h2 += learning_grate * M2 * hidden_input1.Transpose();
                ////
                double[,] array3 = new double[hidden_input1.RowCount, hidden_input1.ColumnCount];
                for (int i = 0; i < hidden_input1.RowCount; i++)
                    for (int j = 0; j < hidden_input1.ColumnCount; j++)
                        array3[i, j] = hidden_errors1[i, j] * hidden_input1[i, j] * (1 - hidden_input1[i, j]);
                Matrix<double> M3 = DenseMatrix.OfArray(array3);
                wih1 += learning_grate * M3 * inputs.Transpose();
                return final_input.ToArray();
            }

            public double[,] query(double[,] inputs_d)
            {
                //расчитать входящие сигналы для скрытого слоя
                //расчитать исходящие сигналы скрытого 
                Matrix<double> inputs = DenseMatrix.OfArray(inputs_d);
                Matrix<double> hidden_input1 = wih1 * inputs;
                for (int i = 0; i < hidden_input1.RowCount; i++)
                    for (int j = 0; j < hidden_input1.ColumnCount; j++)
                        hidden_input1[i, j] = 1 / (1 + Math.Exp(-hidden_input1[i, j])); // V1 = 1 / (1 + exp^(V2)) // hidden_input -> hidden_output //sigma activation
                Matrix<double> hidden_input2 = wh1h2 * hidden_input1;
                for (int i = 0; i < hidden_input2.RowCount; i++)
                    for (int j = 0; j < hidden_input2.ColumnCount; j++)
                        hidden_input2[i, j] = 1 / (1 + Math.Exp(-hidden_input2[i, j])); // V1 = 1 / (1 + exp^(V2)) // hidden_input -> hidden_output //sigma activation
                                                                                        ///////
                                                                                        //расчитать входящие сигналы для выходного слоя
                                                                                        //расчитать исходящие сигналы выходного слоя 
                Matrix<double> final_input = wh2o * hidden_input2;
                for (int i = 0; i < final_input.RowCount; i++)
                    for (int j = 0; j < final_input.ColumnCount; j++)
                        final_input[i, j] = 1 / (1 + Math.Exp(-final_input[i, j])); // V1 = 1 / (1 + exp^(V2)) // final_input -> final_output //sigma activation   
                return final_input.ToArray();

            }
            public void save(string adress)
            {
                using (StreamWriter sw = new StreamWriter(adress, false, System.Text.Encoding.Default))
                {
                    for (int i = 0; i < hidden_nodes_1; i++)
                        for (int j = 0; j < input_nodes; j++)
                        {
                            sw.WriteLine(wih1[i, j]);
                        }
                    sw.WriteLine();
                    //
                    for (int i = 0; i < hidden_nodes_2; i++)
                        for (int j = 0; j < hidden_nodes_1; j++)
                        {
                            sw.WriteLine(wh1h2[i, j]);
                        }
                    sw.WriteLine();
                    //
                    for (int i = 0; i < output_nodes; i++)
                        for (int j = 0; j < hidden_nodes_2; j++)
                        {
                            sw.WriteLine(wh2o[i, j]);
                        }
                    //
                }

            }
        }
   
         static void cut_empty(ref int x, ref int y, int min_y, int min_x, ref int[,] unknow_figure, double m_1, double m_2, bool stop)
        {
            int num_y_del = 0;
            for (int y1 = 0; y1 < y; y1++)
            {
                int mono_line = 0;
                for (int x1 = 0; x1 < x; x1++)
                    if (unknow_figure[x1, y1] != 8)
                        mono_line++;
                if (mono_line >= x * (1 - m_1)) // 0.98    0.02   || mono_line <= x * m_1
                {
                    unknow_figure[0, y1] = 101;
                    num_y_del++;
                }
                else
                {
                    if (stop)
                    {
                        break;
                    }
                }
            }
            if (stop)
                for (int y1 = y - 1; y1 > -1; y1--)
                {
                    int mono_line = 0;
                    for (int x1 = 0; x1 < x; x1++)
                        if (unknow_figure[x1, y1] != 8)
                            mono_line++;
                    if (mono_line >= x * (1 - m_1) || mono_line <= x * m_1) // 0.98    0.02
                    {
                        unknow_figure[0, y1] = 101;
                        num_y_del++;
                    }
                    else
                        break;
                }
            int n_y = y - num_y_del;
            if (n_y <= min_y)
            {
                y = 0;
                return;
            }
            int[,] unknown_figure2 = new int[x, n_y];
            for (int y1 = 0, y2 = 0; y2 < n_y && y1 < y; y1++)
            {
                if (unknow_figure[0, y1] == 101)
                {
                    continue;
                }
                for (int x1 = 0; x1 < x; x1++)
                {
                    unknown_figure2[x1, y2] = unknow_figure[x1, y1];
                }
                y2++;
            }
            unknow_figure = unknown_figure2;
            y = n_y;
            ///

            int num_x_del = 0;
            for (int x1 = 0; x1 < x; x1++)
            {
                int mono_line = 0;
                for (int y1 = 0; y1 < y; y1++)
                    if (unknow_figure[x1, y1] != 8 && unknow_figure[x1, y1] != 4)
                        mono_line++;
                if (mono_line >= y * (1 - m_2))  // 0.95    0.05   || mono_line <= y * m_2
                {
                    unknow_figure[x1, 0] = 100;
                    num_x_del++;
                }
                else
                {
                    if (stop)
                    {
                        break;
                    }
                }
            }
            if (stop)
                for (int x1 = x - 1; x1 > -1; x1--)
                {
                    int mono_line = 0;
                    for (int y1 = 0; y1 < y; y1++)
                        if (unknow_figure[x1, y1] != 8 && unknow_figure[x1, y1] != 4)
                            mono_line++;
                    if (mono_line >= y * (1 - m_2) || mono_line <= y * m_2)  // 0.95    0.05
                    {
                        unknow_figure[x1, 0] = 100;
                        num_x_del++;
                    }
                    else
                        break;
                }
            //
            int n_x = x - num_x_del;
            unknown_figure2 = new int[n_x, y];
            for (int y1 = 0; y1 < y; y1++)
            {
                for (int x1 = 0, x2 = 0; x2 < n_x && x1 < x; x1++)
                {
                    if (unknow_figure[x1, 0] == 100)
                    {
                        continue;
                    }
                    unknown_figure2[x2, y1] = unknow_figure[x1, y1];
                    x2++;
                }
            }
            unknow_figure = unknown_figure2;
            x = n_x;
        }

        static void cut_lines(int x, int y, int[,] unknow_figure, double m_1, int[,,] lines_array, int h_length, double kef_x, int br_lim, int line_length_lim, int[] latitude_array)
        {
            int num_y_del = 0;
            for (int y1 = 0; y1 < y; y1++)
            {
                int mono_line = 0;
                for (int x1 = 0; x1 < x * kef_x; x1++)
                    if (unknow_figure[x1, y1] != 8)
                        mono_line++;
                if (mono_line >= x * kef_x * (1 - m_1) || mono_line <= x * kef_x * m_1) // 0.98    0.02  
                {
                    unknow_figure[0, y1] = 101;
                    num_y_del++;
                }
            }
            //
            int begin_coord = 0, line_length = 0, br = 0, c = 0;
            for (int y1 = 0; y1 < y; y1++)
            {
                if (unknow_figure[0, y1] != 101)
                {
                    if (begin_coord == 0)
                        begin_coord = y1;
                    line_length++;
                    br = 0;
                }
                else
                    br++;
                if (br > br_lim)
                {
                    if (line_length > line_length_lim)
                    {
                        int height = y1 - 1 - br_lim - begin_coord;
                        int[,] unknown_figure2 = new int[x, height];
                        for (int y2 = begin_coord, h = 0; y2 < y1 - br_lim - 1; y2++, h++)
                            for (int x1 = 0; x1 < x; x1++)
                                unknown_figure2[x1, h] = unknow_figure[x1, y2];
                        if (height > h_length)
                            cut_jpg(ref x, ref height, h_length, x, ref unknown_figure2);
                        else
                            add_pixel(ref x, ref height, h_length, x, ref unknown_figure2);
                        for (int i = 0; i < h_length; i++)
                            for (int j = 0; j < x; j++)
                                lines_array[c, j, i] = unknown_figure2[j, i];
                        latitude_array[c] = (y1 - br_lim - 1 - begin_coord) / 2 + begin_coord;
                        c++;
                    }
                    line_length = 0;
                    br = 0;
                    begin_coord = 0;
                }
            }
        }

        static void delete_quadrilaterals(ref int x, ref int y, ref int[,] pixel_array, double kef_x)
        {
            int num_y_del = 0;
            for (int y1 = 0; y1 < y; y1++)
            {
                int snake_line = 0, white_pixel = 0, longest_snake = 0;
                for (int x1 = 0; x1 < x * kef_x; x1++)
                {
                    if (pixel_array[x1, y1] != 8)
                        white_pixel++;
                    else
                    {
                        white_pixel = 0;
                        snake_line++;
                    }
                    if (white_pixel > 6 || x1 == x * kef_x - 1)
                    {
                        white_pixel = 0;
                        if (longest_snake < snake_line)
                        {
                            longest_snake = snake_line;
                            snake_line = 0;
                        }
                    }
                }
                if (longest_snake >= x * 0.1 * kef_x)
                {
                    pixel_array[0, y1] = 101;
                    num_y_del++;
                }
            }
            int n_y = y - num_y_del;
            int[,] unknown_figure2 = new int[x, n_y];
            for (int y1 = 0, y2 = 0; y2 < n_y && y1 < y; y1++)
            {
                if (pixel_array[0, y1] == 101)
                {
                    continue;
                }
                for (int x1 = 0; x1 < x; x1++)
                {
                    unknown_figure2[x1, y2] = pixel_array[x1, y1];
                }
                y2++;
            }
            pixel_array = unknown_figure2;
            y = n_y;

            //
            int num_x_del = 0;
            for (int x1 = 0; x1 < x * kef_x; x1++)
            {
                int mono_line = 0;
                for (int y1 = 0; y1 < y; y1++)
                    if (pixel_array[x1, y1] != 8)
                        mono_line++;
                if (mono_line <= y * 0.4)  // 0.95    0.05
                {
                    pixel_array[x1, 0] = 100;
                    num_x_del++;
                }
            }
            //
            int n_x = x - num_x_del;
            unknown_figure2 = new int[n_x, y];
            for (int y1 = 0; y1 < y; y1++)
            {
                for (int x1 = 0, x2 = 0; x2 < n_x && x1 < x; x1++)
                {
                    if (pixel_array[x1, 0] == 100)
                    {
                        continue;
                    }
                    unknown_figure2[x2, y1] = pixel_array[x1, y1];
                    x2++;
                }
            }
            pixel_array = unknown_figure2;
            x = n_x;

        }
        static void add_pixel(ref int x, ref int y, int max_y, int max_x, ref int[,] unknow_figure)
        {
            int x0 = max_x;
            int y0 = max_y;

            double add_piksel = x0 - x;
            if (add_piksel > 0)
            {
                int[,] unknown_figure2 = new int[x0, y];
                for (int y1 = 0; y1 < y; y1++)
                {
                    add_piksel = x0 - x;
                    double length_add = Convert.ToDouble(max_x) / add_piksel;
                    double adder = length_add;
                    for (int x1 = 0, x2 = 0; x1 < x; x1++, x2++)
                    {
                        unknown_figure2[x2, y1] = unknow_figure[x1, y1];
                        if (x2 == Convert.ToInt16(adder) && add_piksel > 0)
                        {
                            x2++;
                            unknown_figure2[x2, y1] = unknow_figure[x1, y1];
                            adder = adder + length_add;
                            add_piksel--;
                        }
                    }
                    if (add_piksel > 0)
                        unknown_figure2[x0 - 1, y1] = unknow_figure[x - 1, y1];
                }
                unknow_figure = unknown_figure2;
                x = x0;
            }
            add_piksel = y0 - y;
            if (add_piksel > 0)
            {
                int[,] unknown_figure2 = new int[x0, y0];
                for (int x1 = 0; x1 < x; x1++)
                {
                    add_piksel = y0 - y;
                    double length_add = Convert.ToDouble(max_y) / add_piksel;
                    double adder = length_add;
                    for (int y1 = 0, y2 = 0; y1 < y; y1++, y2++)
                    {
                        unknown_figure2[x1, y2] = unknow_figure[x1, y1];
                        if (y2 == Convert.ToInt16(adder) && add_piksel > 0)
                        {
                            y2++;
                            unknown_figure2[x1, y2] = unknow_figure[x1, y1];
                            adder = adder + length_add;
                            add_piksel--;
                        }
                    }
                    if (add_piksel > 0)
                        unknown_figure2[x1, y0 - 1] = unknow_figure[x1, y - 1];
                }
                unknow_figure = unknown_figure2;
                y = y0;
            }

        }
        static void cut_jpg(ref int x, ref int y, int min_y, int min_x, ref int[,] unknow_figure)
        {
            //
            //cut_empty(ref x, ref y, min_y, min_x, ref unknow_figure, 0.02, 0.05, false);
            //if (x <= min_x || y <= min_y)
            //    return;
            //
            double x_d = x;
            double y_d = y;

            int x0 = min_x;
            int y0 = min_y;

            double cut_piksel = x - x0;
            if (cut_piksel > 0)
            {
                double length_cut = x_d / cut_piksel;
                double deleter = length_cut;
                for (int x1 = 0; x1 < x; x1++)
                {
                    if (x1 == Convert.ToInt16(deleter) && cut_piksel > 0)
                    {
                        unknow_figure[x1, 0] = 100;
                        deleter = deleter + length_cut;
                        cut_piksel--;
                    }
                }
                int[,] unknown_figure2 = new int[x0, y];
                for (int y1 = 0; y1 < y; y1++)
                {
                    for (int x1 = 0, x2 = 0; x2 < x0 && x1 < x; x1++)
                    {
                        if (unknow_figure[x1, 0] == 100)
                        {
                            continue;
                        }
                        unknown_figure2[x2, y1] = unknow_figure[x1, y1];
                        x2++;
                    }
                }
                unknow_figure = unknown_figure2;
                x = x0;
            }
            cut_piksel = y - y0;
            if (cut_piksel > 0)
            {
                double length_cut = y / cut_piksel;
                double deleter = length_cut;
                for (int y1 = 0; y1 < y; y1++)
                {
                    if (y1 == Convert.ToInt16(deleter) && cut_piksel > 0)
                    {
                        unknow_figure[0, y1] = 101;
                        deleter = deleter + length_cut;
                        cut_piksel--;
                    }
                }
                int[,] unknown_figure2 = new int[x0, y0];
                for (int y1 = 0, y2 = 0; y2 < y0 && y1 < y; y1++)
                {
                    if (unknow_figure[0, y1] == 101)
                    {
                        continue;
                    }
                    for (int x1 = 0; x1 < x0 && x1 < x; x1++)
                    {
                        unknown_figure2[x1, y2] = unknow_figure[x1, y1];
                    }
                    y2++;
                }
                unknow_figure = unknown_figure2;
                y = y0;
            }
        }
        static void Int_palette(Bitmap bmpImg, int cube_root_of_colors, int[,] pixel_int_array)
        {

            Color color = bmpImg.GetPixel(0, 0);
            //
            double d_palette_size = Math.Pow(cube_root_of_colors, 3);
            int palette_size = Convert.ToInt16(d_palette_size);

            int[] diapason = new int[cube_root_of_colors];
            for (int z = 0, f = 0; z < color.A - color.A / cube_root_of_colors / 2; z = z + color.A / cube_root_of_colors, f++) // КОСТІЛЬ
            {
                diapason[f] = z;
            }
            //
            int[,] i_color = new int[palette_size, 3];
            int palette_size_copy = palette_size - 1;
            for (int i = 0; i < cube_root_of_colors; i++)
            {
                for (int z = 0; z < cube_root_of_colors; z++)
                {
                    for (int j = 0; j < cube_root_of_colors; j++)
                    {
                        i_color[palette_size_copy, 0] = i;
                        i_color[palette_size_copy, 1] = z;
                        i_color[palette_size_copy, 2] = j;
                        palette_size_copy--;
                    }
                }
            }
            //
            for (int j = 0; j < bmpImg.Height; j++)
            {
                for (int i = 0; i < bmpImg.Width; i++)
                {
                    color = bmpImg.GetPixel(i, j);
                    int[] pixsel_rgb_adapt = new int[3];
                    for (int p = 0; p < cube_root_of_colors; p++)
                    {
                        if (color.R > diapason[p])
                        {
                            pixsel_rgb_adapt[0] = p;
                        }
                        if (color.G > diapason[p])
                        {
                            pixsel_rgb_adapt[1] = p;
                        }
                        if (color.B > diapason[p])
                        {
                            pixsel_rgb_adapt[2] = p;
                        }
                    }
                    int number_color = 0;
                    for (int z = 0; z < palette_size; z++)
                    {
                        if (i_color[z, 0] == pixsel_rgb_adapt[0] && i_color[z, 1] == pixsel_rgb_adapt[1] && i_color[z, 2] == pixsel_rgb_adapt[2])
                        {
                            number_color = z + 1;
                            break;
                        }
                    }
                    pixel_int_array[i, j] = number_color;
                }
            }
        }

        static void Perimeter(int x, int y, ref int[,] pixel_array, int sw, ref int[,,] dev_digit_array, ref int num_digit, int lenght_squre)
        {
            int[,] original_array = new int[x, y];
            for (int y1 = 0; y1 < y; y1++)
                for (int x1 = 0; x1 < x; x1++)
                    original_array[x1, y1] = pixel_array[x1, y1];
            int[,] pixel_array_copy = new int[x, y];
            List<List<int>> coordinate_x = new List<List<int>>();
            coordinate_x.Add(new List<int>());
            coordinate_x.Add(new List<int>());

            List<List<int>> coordinate_y = new List<List<int>>();
            coordinate_y.Add(new List<int>());
            coordinate_y.Add(new List<int>());
            List<int> metr_list = new List<int>();
            int endep_board = 1;
            for (int y1 = 0; y1 < y; y1++)
                for (int x1 = 0; x1 < x; x1++)
                {
                    //
                    int pixel = pixel_array[x1, y1];
                    if (pixel < 0 || pixel != 8)
                        continue;

                    int y2 = y1 - 1, y0 = y1, x0 = x1, meter = 0;
                    for (; y2 < y0 + 2 && y2 < y; y2++)
                    {
                        if (y2 < 0)
                            y2 = 0;
                        int x2 = x0 - 1;
                        if (x2 < 0)
                            x2 = 0;
                        for (; x2 < x0 + 2 && x2 < x; x2++)
                        {
                            if (x2 == x0 && y2 == y0)
                                continue;
                            if (pixel_array[x2, y2] == pixel)
                            {
                                ////
                                int y3 = y2 - 1;
                                if (y3 < 0)
                                    y3 = 0;
                                int another_color = 0;
                                for (; y3 < y2 + 2 && y3 < y; y3++)
                                {
                                    int x3 = x2 - 1;
                                    if (x3 < 0)
                                        x3 = 0;
                                    for (; x3 < x2 + 2 && x3 < x; x3++)
                                    {
                                        if (x3 == x2 && y3 == y2)
                                            continue;
                                        if (pixel_array[x3, y3] != pixel && pixel_array[x3, y3] != -pixel || x3 == 0 || x3 == x - 1 || y3 == 0 || y3 == y - 1)
                                            another_color++;
                                    }
                                }
                                if (another_color > 0) // && another_color < 4) // ??????????
                                {
                                    pixel_array[x0, y0] = -pixel;
                                    pixel_array_copy[x0, y0] = endep_board;
                                    coordinate_x[endep_board].Add(x0);
                                    coordinate_y[endep_board].Add(y0);
                                    meter++;
                                    y0 = y2;
                                    x0 = x2;
                                    y2 = y0 - 2;
                                    break;
                                }

                            }
                        }
                    }
                    if (meter > 0)
                    {
                        metr_list.Add(meter);
                        coordinate_x.Add(new List<int>());
                        coordinate_y.Add(new List<int>());
                        endep_board++;
                    }
                }

            //
            List<List<int>> chains = new List<List<int>>();
            for (int y1 = 0; y1 < y; y1++)
                for (int x1 = 0; x1 < x; x1++)
                {
                    //
                    int pixel = pixel_array[x1, y1];
                    if (pixel > 0)
                        continue;
                    int y2 = y1 - 1;
                    if (y2 < 0)
                        y2 = 0;
                    for (; y2 < y1 + 2 && y2 < y; y2++)
                    {
                        int x2 = x1 - 1;
                        if (x2 < 0)
                            x2 = 0;
                        for (; x2 < x1 + 2 && x2 < x; x2++)
                        {
                            if (x2 == x1 && y2 == y1)
                                continue;
                            int fig_copy_1 = pixel_array_copy[x1, y1];
                            int fig_copy_2 = pixel_array_copy[x2, y2];
                            if (pixel_array[x2, y2] == pixel && fig_copy_2 != fig_copy_1 && fig_copy_1 < metr_list.Count && fig_copy_2 < metr_list.Count)
                            {
                                bool exists_1 = false;
                                int chain_1_number = 0;
                                bool exists_2 = false;
                                int chain_2_number = 0;
                                for (int i = 0; i < chains.Count; i++)
                                    if (chains[i][0] > 0)
                                    {
                                        for (int j = 5; j < chains[i].Count; j++)
                                            if (chains[i][j] == fig_copy_1)
                                            {
                                                exists_1 = true;
                                                chain_1_number = i;
                                                break;
                                            }
                                        if (exists_1)
                                            break;
                                    }
                                for (int i = 0; i < chains.Count; i++)
                                    if (chains[i][0] > 0)
                                    {
                                        for (int j = 5; j < chains[i].Count; j++)
                                            if (chains[i][j] == fig_copy_2)
                                            {
                                                exists_2 = true;
                                                chain_2_number = i;
                                                break;
                                            }
                                        if (exists_2)
                                            break;
                                    }

                                if (chain_1_number == chain_2_number && exists_1 && exists_2)
                                    continue;
                                if (exists_1 && exists_2)
                                {
                                    if (chains[chain_1_number].Count > chains[chain_2_number].Count)
                                    {
                                        for (int i = 5; i < chains[chain_2_number].Count; i++)
                                            chains[chain_1_number].Add(chains[chain_2_number][i]);

                                        int max_x2 = chains[chain_2_number][0];
                                        if (chains[chain_1_number][0] < max_x2)
                                            chains[chain_1_number][0] = max_x2;
                                        int min_x2 = chains[chain_2_number][1];
                                        if (chains[chain_1_number][1] > min_x2)
                                            chains[chain_1_number][1] = min_x2;
                                        int max_y2 = chains[chain_2_number][2];
                                        if (chains[chain_1_number][2] < max_y2)
                                            chains[chain_1_number][2] = max_y2;
                                        int min_y2 = chains[chain_2_number][3];
                                        if (chains[chain_1_number][3] > min_y2)
                                            chains[chain_1_number][3] = min_y2;
                                        chains[chain_1_number][4] = chains[chain_1_number][4] + chains[chain_2_number][4];
                                        chains[chain_2_number][0] = -1;
                                    }
                                    else
                                    {
                                        for (int i = 5; i < chains[chain_1_number].Count; i++)
                                            chains[chain_2_number].Add(chains[chain_1_number][i]);

                                        int max_x2 = chains[chain_1_number][0];
                                        if (chains[chain_2_number][0] < max_x2)
                                            chains[chain_2_number][0] = max_x2;
                                        int min_x2 = chains[chain_1_number][1];
                                        if (chains[chain_2_number][1] > min_x2)
                                            chains[chain_2_number][1] = min_x2;
                                        int max_y2 = chains[chain_1_number][2];
                                        if (chains[chain_2_number][2] < max_y2)
                                            chains[chain_2_number][2] = max_y2;
                                        int min_y2 = chains[chain_1_number][3];
                                        if (chains[chain_2_number][3] > min_y2)
                                            chains[chain_2_number][3] = min_y2;
                                        chains[chain_2_number][4] = chains[chain_2_number][4] + chains[chain_1_number][4];

                                        chains[chain_1_number][0] = -1;
                                    }

                                }
                                if (exists_1 == false && exists_2 == false)
                                {
                                    chains.Add(new List<int>());
                                    int num = chains.Count - 1;

                                    int max_x1 = coordinate_x[fig_copy_1].Max();
                                    int max_x2 = coordinate_x[fig_copy_2].Max();
                                    if (max_x1 > max_x2)
                                        chains[num].Add(max_x1);
                                    else
                                        chains[num].Add(max_x2);
                                    int min_x1 = coordinate_x[fig_copy_1].Min();
                                    int min_x2 = coordinate_x[fig_copy_2].Min();
                                    if (min_x1 < min_x2)
                                        chains[num].Add(min_x1);
                                    else
                                        chains[num].Add(min_x2);

                                    int max_y1 = coordinate_y[fig_copy_1].Max();
                                    int max_y2 = coordinate_y[fig_copy_2].Max();
                                    if (max_y1 > max_y2)
                                        chains[num].Add(max_y1);
                                    else
                                        chains[num].Add(max_y2);
                                    int min_y1 = coordinate_y[fig_copy_1].Min();
                                    int min_y2 = coordinate_y[fig_copy_2].Min();
                                    if (min_y1 < min_y2)
                                        chains[num].Add(min_y1);
                                    else
                                        chains[num].Add(min_y2);
                                    chains[num].Add(metr_list[fig_copy_1] + metr_list[fig_copy_2]);
                                    chains[num].Add(fig_copy_1);
                                    chains[num].Add(fig_copy_2);
                                }
                                if (exists_1 && exists_2 == false)
                                {
                                    chains[chain_1_number].Add(fig_copy_2);
                                    int max_x2 = coordinate_x[fig_copy_2].Max();
                                    if (chains[chain_1_number][0] < max_x2)
                                        chains[chain_1_number][0] = max_x2;
                                    int min_x2 = coordinate_x[fig_copy_2].Min();
                                    if (chains[chain_1_number][1] > min_x2)
                                        chains[chain_1_number][1] = min_x2;
                                    int max_y2 = coordinate_y[fig_copy_2].Max();
                                    if (chains[chain_1_number][2] < max_y2)
                                        chains[chain_1_number][2] = max_y2;
                                    int min_y2 = coordinate_y[fig_copy_2].Min();
                                    if (chains[chain_1_number][3] > min_y2)
                                        chains[chain_1_number][3] = min_y2;
                                    chains[chain_1_number][4] = chains[chain_1_number][4] + metr_list[fig_copy_2];
                                }
                                if (exists_1 == false && exists_2)
                                {
                                    chains[chain_2_number].Add(fig_copy_1);
                                    int max_x2 = coordinate_x[fig_copy_1].Max();
                                    if (chains[chain_2_number][0] < max_x2)
                                        chains[chain_2_number][0] = max_x2;
                                    int min_x2 = coordinate_x[fig_copy_1].Min();
                                    if (chains[chain_2_number][1] > min_x2)
                                        chains[chain_2_number][1] = min_x2;
                                    int max_y2 = coordinate_y[fig_copy_1].Max();
                                    if (chains[chain_2_number][2] < max_y2)
                                        chains[chain_2_number][2] = max_y2;
                                    int min_y2 = coordinate_y[fig_copy_1].Min();
                                    if (chains[chain_2_number][3] > min_y2)
                                        chains[chain_2_number][3] = min_y2;
                                    chains[chain_2_number][4] = chains[chain_2_number][4] + metr_list[fig_copy_1];
                                }

                            }
                        }
                    }
                }
            if (sw == 1)
            {
                for (int z = 0; z < chains.Count; z++)
                    if (chains[z][4] < 10 || chains[z][0] < 0)
                    {
                        chains.RemoveAt(z);
                        z--;
                    }
                chains.Add(new List<int>());
                chains[chains.Count - 1] = chains[0];
                for (int i = 0; i < y; i++)
                    for (int j = 0; j < x; j++)
                    {
                        if (pixel_array[j, i] > 0)
                            continue;
                        int fig_copy = pixel_array_copy[j, i];
                        bool a = false;
                        for (int z = 1; z < chains.Count; z++)
                        {
                            for (int t = 5; t < chains[z].Count; t++)
                                if (fig_copy == chains[z][t])
                                {
                                    a = true;
                                    pixel_array_copy[j, i] = -z;
                                    break;
                                }
                            if (a)
                                break;
                        }
                    }
                for (int i = 0; i < y; i++)
                {
                    // string st = "";
                    for (int j = 0; j < x; j++)
                    {
                        if (pixel_array_copy[j, i] < 0)
                        {
                            // st = st + 0;
                            pixel_array_copy[j, i] = 8;
                            continue;
                        }
                        if (pixel_array[j, i] == 4)
                        {
                            //st = st + 1;
                            pixel_array_copy[j, i] = 4;
                            continue;
                        }
                        pixel_array_copy[j, i] = 1;
                        // st = st + " ";
                    }
                    //  System.Diagnostics.Debug.WriteLine(st);
                }
                // System.Diagnostics.Debug.WriteLine("_____________");
                for (int y1 = 0; y1 < y; y1++)
                    for (int x1 = 0; x1 < x; x1++)
                    {
                        //
                        int pixel = pixel_array_copy[x1, y1];
                        if (pixel != 1)
                            continue;

                        int y2 = y1 - 1, blacky = 0;
                        for (; y2 < y1 + 2 && y2 < y; y2++)
                        {
                            if (y2 < 0)
                                y2 = 0;
                            int x2 = x1 - 1;
                            if (x2 < 0)
                                x2 = 0;
                            for (; x2 < x1 + 2 && x2 < x; x2++)
                            {
                                if (pixel_array_copy[x2, y2] == 8)
                                    blacky++;
                            }
                        }
                        if (blacky > 4)
                            pixel_array_copy[x1, y1] = 8;
                    }
                ///
                pixel_array = pixel_array_copy;
                //for (int z = 1; z < chains.Count; z++)
                //{
                //    for (int i = chains[z][3]; i < chains[z][2] + 1; i++)
                //    {
                //        string st = "";
                //        for (int j = chains[z][1]; j < chains[z][0] + 1; j++)
                //        {
                //            if (pixel_array_copy[j, i] == -z)
                //                st = st + 0;
                //            else
                //                st = st + 1;
                //        }
                //        System.Diagnostics.Debug.WriteLine.WriteLine(st);
                //    }

                //}
            }
            if (sw == 2)
            {
                for (int z = 0; z < chains.Count; z++)
                {
                    bool gate = chains[z][4] < 50 && chains[z][2] - chains[z][3] < y * 0.3 && chains[z][0] - chains[z][1] < x * 0.05;
                    if (chains[z][0] < 0 || gate)
                    {
                        chains.RemoveAt(z);
                        z--;
                    }
                }
                // sort 
                List<List<int>> chains_sort = new List<List<int>>();
                for (int i = 0; i < chains.Count; i++)
                {
                    int min_x = x;
                    int num = -1;
                    for (int z = 0; z < chains.Count; z++)
                    {
                        if (chains[z][4] > -1 && min_x > chains[z][1])
                        {
                            min_x = chains[z][1];
                            num = z;
                        }
                    }
                    chains_sort.Add(chains[num]);
                    chains[num][4] = -1;
                }
                //
                chains = chains_sort;
                double kef = 0.25;
                for (int i = 0; i < chains.Count; i++)
                    for (int z = 0; z < chains.Count; z++)
                        if (i != z && chains[i][4] != -2 && chains[z][4] != -2)
                        {
                            int beg_z = chains[z][1], beg_i = chains[i][1], end_z = chains[z][0], end_i = chains[i][0];
                            if (beg_z >= beg_i && end_z <= end_i)
                            {
                                double per = Convert.ToDouble(end_z - beg_z) / Convert.ToDouble(end_i - beg_i);  // ????
                                if (per > kef / 2)
                                    chains[z][4] = -2;
                                continue;
                            }
                            if (beg_z <= beg_i && end_z <= end_i && end_z > beg_i - 10)
                            {
                                if (end_z - beg_z < end_i - beg_i)
                                {
                                    double per = Convert.ToDouble(end_z - beg_z) / Convert.ToDouble(end_i - beg_i);  // ????

                                    if (per > kef &&
                            Convert.ToDouble(chains[i][0] - chains[z][1]) / Convert.ToDouble(x) < 0.2)
                                    {
                                        chains[z][4] = -2;
                                        chains[i][1] = chains[z][1];
                                    }
                                    continue;
                                }
                                else
                                {
                                    double per = Convert.ToDouble(end_i - beg_i) / Convert.ToDouble(end_z - beg_z);  // ????
                                    if (per > kef &&
                            Convert.ToDouble(chains[i][0] - chains[z][1]) / Convert.ToDouble(x) < 0.2)
                                    {
                                        chains[i][4] = -2;
                                        chains[z][0] = chains[i][0];
                                    }

                                    continue;
                                }
                            }
                            if (beg_z >= beg_i && end_z >= end_i && beg_z - 10 < end_i)
                            {
                                if (end_z - beg_z < end_i - beg_i)
                                {
                                    double per = Convert.ToDouble(end_z - beg_z) / Convert.ToDouble(end_i - beg_i);  // ????
                                    if (per > kef &&
                            Convert.ToDouble(chains[z][0] - chains[i][1]) / Convert.ToDouble(x) < 0.2)
                                    {
                                        chains[z][4] = -2;
                                        chains[i][0] = chains[z][0];
                                    }
                                    continue;
                                }
                                else
                                {
                                    double per = Convert.ToDouble(end_i - beg_i) / Convert.ToDouble(end_z - beg_z);  // ????
                                    if (per > kef &&
                            Convert.ToDouble(chains[z][0] - chains[i][1]) / Convert.ToDouble(x) < 0.2)
                                    {
                                        chains[i][4] = -2;
                                        chains[z][1] = chains[i][1];
                                    }
                                    continue;
                                }
                            }
                        }


                //for (int i = 0; i < y; i++)
                //{
                //    string st = "";
                //    for (int j = 0; j < x; j++)
                //    {
                //        if (original_array[j, i] == 8)
                //            st = st + 0;
                //        else
                //            st = st + " ";
                //    }
                //    System.Diagnostics.Debug.WriteLine(st);
                //}
                //System.Diagnostics.Debug.WriteLine("_____________");
              
                //  int lenght_squre = 28;
                num_digit = 0;
                for (int z = 0; z < chains.Count; z++)
                    if (chains[z][4] != -2 && chains[z][0] - chains[z][1] > x * 0.05)  // warning
                    {
                        double dev = Convert.ToDouble(chains[z][0] - chains[z][1]) / Convert.ToDouble(x);
                        if (dev < 0.2)
                        {
                            for (int i = 0; i < y; i++)
                            {
                                string st = "";
                                for (int j = chains[z][1]; j < chains[z][0] + 1; j++)
                                {
                                    if (original_array[j, i] == 8)
                                        st = st + 0;
                                    else
                                        st = st + " ";
                                }
                                //.. System.Diagnostics.Debug.WriteLine(st);
                            }
                            num_digit++;
                        }
                        else
                        {
                            int half = (chains[z][0] - chains[z][1]) / 2;
                            for (int i = 0; i < y; i++)
                            {
                                string st = "";
                                for (int j = chains[z][1]; j < chains[z][1] + half; j++)
                                {
                                    if (original_array[j, i] == 8)
                                        st = st + 0;
                                    else
                                        st = st + " ";
                                }
                                // System.Diagnostics.Debug.WriteLine(st);
                            }
                            for (int i = 0; i < y; i++)
                            {
                                string st = "";
                                for (int j = chains[z][1] + half; j < chains[z][0] + 1; j++)
                                {
                                    if (original_array[j, i] == 8)
                                        st = st + 0;
                                    else
                                        st = st + " ";
                                }
                                //  System.Diagnostics.Debug.WriteLine(st);
                            }
                            num_digit = num_digit + 2;
                        }
                        //  System.Diagnostics.Debug.WriteLine(dev);
                    }
                dev_digit_array = new int[num_digit, lenght_squre, lenght_squre];

                num_digit = 0;
                for (int z = 0; z < chains.Count; z++)
                    if (chains[z][4] != -2 && chains[z][0] - chains[z][1] > x * 0.05)
                    {
                        double dev = Convert.ToDouble(chains[z][0] - chains[z][1]) / Convert.ToDouble(x);
                        if (dev < 0.2)
                        {
                            int x_copy = chains[z][0] - chains[z][1], y_copy = y;
                            int[,] array1 = new int[x_copy, y_copy];
                            for (int i = 0; i < y; i++)
                                for (int j = chains[z][1], width = 0; j < chains[z][0] + 1 && width < x_copy; j++, width++)
                                    array1[width, i] = original_array[j, i];

                            cut_empty(ref x_copy, ref y_copy, 0, 0, ref array1, 0.05, 0.05, false);
                            cut_jpg(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);
                            add_pixel(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);

                            for (int i = 0; i < y_copy; i++)
                                for (int j = 0; j < x_copy; j++)
                                    dev_digit_array[num_digit, j, i] = array1[j, i];
                            num_digit++;

                        }
                        else
                        {
                            int half = (chains[z][0] - chains[z][1]) / 2;

                            int x_copy = half, y_copy = y;
                            int[,] array1 = new int[x_copy, y_copy];
                            for (int i = 0; i < y; i++)
                                for (int j = chains[z][1], width = 0; j < chains[z][1] + half && width < x_copy; j++, width++)
                                    array1[width, i] = original_array[j, i];
                            cut_empty(ref x_copy, ref y_copy, 0, 0, ref array1, 0.05, 0.05, false);
                            cut_jpg(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);
                            add_pixel(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);
                            for (int i = 0; i < y_copy; i++)
                                for (int j = 0; j < x_copy; j++)
                                    dev_digit_array[num_digit, j, i] = array1[j, i];
                            num_digit++;

                            x_copy = half; y_copy = y;
                            array1 = new int[x_copy, y_copy];
                            for (int i = 0; i < y; i++)
                                for (int j = chains[z][1] + half, width = 0; j < chains[z][0] + 1 && width < x_copy; j++, width++)
                                    array1[width, i] = original_array[j, i];
                            cut_empty(ref x_copy, ref y_copy, 0, 0, ref array1, 0.05, 0.05, false);
                            cut_jpg(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);
                            add_pixel(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);
                            for (int i = 0; i < y_copy; i++)
                                for (int j = 0; j < x_copy; j++)
                                    dev_digit_array[num_digit, j, i] = array1[j, i];
                            num_digit++;
                        }
                    }



                //for (int z = 1; z < chains.Count; z++)
                //{
                //    for (int i = chains[z][3]; i < chains[z][2] + 1; i++)
                //    {
                //        string st = "";
                //        for (int j = chains[z][1]; j < chains[z][0] + 1; j++)
                //        {
                //            if (pixel_array_copy[j, i] == -z)
                //                st = st + 0;
                //            else
                //                st = st + " ";
                //        }
                //        System.Diagnostics.Debug.WriteLine(st);
                //    }
                //}
            }

            if (sw == 3)
            {
                for (int z = 0; z < chains.Count; z++)
                {
                    if (chains[z][0] < 0) //|| chains[z][4] < 3) //warning
                    {
                        chains.RemoveAt(z);
                        z--;
                    }
                }
                // sort 
                List<List<int>> chains_sort = new List<List<int>>();
                for (int i = 0; i < chains.Count; i++)
                {
                    int min_x = x;
                    int num = -1;
                    for (int z = 0; z < chains.Count; z++)
                    {
                        if (chains[z][4] > -1 && min_x > chains[z][1])
                        {
                            min_x = chains[z][1];
                            num = z;
                        }
                    }
                    chains_sort.Add(chains[num]);
                    chains[num][4] = -1;
                }
                //
                chains = chains_sort;

                //for (int i = 0; i < y; i++)
                //{
                //    string st = "";
                //    for (int j = 0; j < x; j++)
                //    {
                //        if (original_array[j, i] == 8)
                //            st = st + 0;
                //        else
                //            st = st + " ";
                //    }
                //    System.Diagnostics.Debug.WriteLine(st);
                //}
                //System.Diagnostics.Debug.WriteLine("_____________");

                //  int lenght_squre = 28;
                num_digit = 0;
                for (int z = 0; z < chains.Count; z++)
                {
                    for (int i = 0; i < y; i++)
                    {
                        string st = "";
                        for (int j = chains[z][1]; j < chains[z][0] + 1; j++)
                        {
                            if (original_array[j, i] == 8)
                                st = st + 0;
                            else
                                st = st + " ";
                        }
                        //.. System.Diagnostics.Debug.WriteLine(st);
                    }
                    num_digit++;
                    //  System.Diagnostics.Debug.WriteLine(dev);
                }

                dev_digit_array = new int[num_digit, lenght_squre, lenght_squre];

                num_digit = 0;
                for (int z = 0; z < chains.Count; z++)
                {
                    if (chains[z][2] - chains[z][3] < y * 0.5)
                    {
                        dev_digit_array[num_digit, 0, 0] = -2;
                        num_digit++;
                        continue;
                    }
                    int x_copy = chains[z][0] - chains[z][1], y_copy = y;
                    int[,] array1 = new int[x_copy, y_copy];
                    for (int i = 0; i < y; i++)
                        for (int j = chains[z][1], width = 0; j < chains[z][0] + 1 && width < x_copy; j++, width++)
                            array1[width, i] = original_array[j, i];

                    cut_jpg(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);
                    add_pixel(ref x_copy, ref y_copy, lenght_squre, lenght_squre, ref array1);

                    for (int i = 0; i < y_copy; i++)
                        for (int j = 0; j < x_copy; j++)
                            dev_digit_array[num_digit, j, i] = array1[j, i];
                    num_digit++;

                }

            }
        }

        static void save_input(int ph_length, int[,,] pixel_array, int length_y, int length_x, string path)
        {
            using (StreamWriter sw = new StreamWriter(path, true, System.Text.Encoding.Default))
            {
                for (int ph = 0; ph < ph_length; ph++)
                {
                    if (pixel_array[ph, 0, 0] == -1)
                        continue;
                    for (int j = 0; j < length_y; j++)
                    {
                        string st = "";
                        for (int i = 0; i < length_x; i++)
                        {
                            if (pixel_array[ph, i, j] == 8)
                            {
                                st = st + 0;
                                continue;
                            }
                            if (pixel_array[ph, i, j] == 4)
                            {
                                st = st + 1;
                                continue;
                            }
                            if (pixel_array[ph, i, j] == 10)
                            {
                                st = st + "-";
                                continue;
                            }
                            st = st + " ";
                        }
                        sw.WriteLine(st);
                    }
                    sw.WriteLine();
                }

            }
        }
        static void read_input(ref int[,,] pixel_array, string st, int length_y, int length_x, ref int num)
        {
            using (StreamReader sr1 = new StreamReader(st))
            {
                List<string> all_lines = new List<string>();
                string line;
                while ((line = sr1.ReadLine()) != null)
                    all_lines.Add(line);
                num = all_lines.Count / (length_y + 1);
                pixel_array = new int[num, length_x, length_y];
                for (int c = 0, c1 = 0; c < all_lines.Count; c++, c1++)
                {
                    for (int i = 0; i < length_y; i++, c++)
                    {
                        char[] arr = all_lines[c].ToCharArray();
                        for (int j = 0; j < length_x; j++)
                        {
                            if (arr[j] == '0')
                            {
                                pixel_array[c1, j, i] = 8;//   pixel_array[c1, j, i] = 0.99; //  
                                continue;
                            }
                            if (arr[j] == ' ')
                            {
                                pixel_array[c1, j, i] = 1; //   pixel_array[c1, j, i] = 0.01; //  
                                continue;
                            }
                            if (arr[j] == '-')
                            {
                                pixel_array[c1, j, i] = -1; //   pixel_array[c1, j, i] = 0.01; //  
                                continue;
                            }
                            pixel_array[c1, j, i] = 4; //  pixel_array[c1, j, i] = -0.99; //   //
                        }
                    }

                }


            }
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            double[,] ex1 = new double[2, 2] { { 1, 2 }, { 0, 1 } };
            Matrix<double> ex2 = DenseMatrix.OfArray(ex1);
            neural_network_2hid[] Intelect2 = new neural_network_2hid[4];



            this.openFileDialog1.Filter = "Images (*.PNG)|*.PNG|" + "All files (*.*)|*.*";
            this.openFileDialog1.Multiselect = true;
            this.openFileDialog1.Title = "Image Browser(Multiselect enabled)";

            DialogResult dr = this.openFileDialog1.ShowDialog();
            string[] photo_address = new string[0];
            if (dr == System.Windows.Forms.DialogResult.OK)
            {
                try
                {
                    photo_address = openFileDialog1.FileNames;
                }
                catch (Exception)
                {
                }
            }

            /// united
            Intelect2[0] = new neural_network_2hid(1000, 200, 75, 1, 0, @"C:\projects\OperatorVision\OperatorVision\txt\weight_2_0.txt");

            Intelect2[1] = new neural_network_2hid(400, 200, 75, 11, 0, @"C:\projects\OperatorVision\OperatorVision\txt\weight_1_0.txt");

            Intelect2[2] = new neural_network_2hid(2000, 200, 75, 1, 0, @"C:\projects\OperatorVision\OperatorVision\txt\weight_3_0.txt");

            Intelect2[3] = new neural_network_2hid(400, 200, 75, 10, 0, @"C:\projects\OperatorVision\OperatorVision\txt\weight_4_0.txt");

            for (int ph = 0; ph < photo_address.Length; ph++)
            {
                using (Image originalImage = Image.FromFile(photo_address[ph]))
                {
                    int x_s1 = originalImage.Width;
                    int y_s1 = originalImage.Height;
                    int[,] pixel_array_s1 = new int[x_s1, y_s1];
                    Int_palette((Bitmap)originalImage, 2, pixel_array_s1);
                    cut_empty(ref x_s1, ref y_s1, 0, 0, ref pixel_array_s1, 0.1, 0.1, true);
                    if (x_s1 != 0 && y_s1 != 0)
                    {
                        int x1_s2 = 185, x2_s2 = 370;
                        int[,] pixel_array1_s2 = new int[x1_s2, y_s1];
                        int[,] pixel_array2_s2 = new int[x2_s2, y_s1];
                        for (int h = 0; h < y_s1; h++)
                        {
                            for (int w = 0; w < x1_s2; w++)
                                pixel_array1_s2[w, h] = pixel_array_s1[w + 5, h];
                            for (int w = 0; w < x2_s2; w++)
                                pixel_array2_s2[w, h] = pixel_array_s1[w + 1700, h];
                        }
                        int h1_s3 = 20;
                        int[,,] lines_array1_s3 = new int[65, x1_s2, h1_s3];
                        int[] latitude_array1_s3 = new int[65];
                        string[,] st_array1 = new string[65, 11];
                        cut_lines(x1_s2, y_s1, pixel_array1_s2, 0.05, lines_array1_s3, h1_s3, 1, 3, 10, latitude_array1_s3);
                        for (int c_s3 = 0; c_s3 < 65; c_s3++)
                        {
                            if (lines_array1_s3[c_s3, 0, 0] != 0)
                            {
                                int x_copy = x1_s2, y_copy = h1_s3;
                                int[,] pixel_array1_s3 = new int[x_copy, y_copy];
                                int[,] pixel_array2_s3 = new int[x_copy, y_copy];
                                for (int i = 0; i < y_copy; i++)
                                    for (int j = 0; j < x_copy; j++)
                                    {
                                        pixel_array1_s3[j, i] = lines_array1_s3[c_s3, j, i];
                                        pixel_array2_s3[j, i] = lines_array1_s3[c_s3, j, i];
                                    }

                                int x_s4 = 100, y_s4 = 10;
                                cut_jpg(ref x_copy, ref y_copy, y_s4, x_s4, ref pixel_array1_s3);

                                ////
                                double[,] inputs = new double[x_s4 * y_s4, 1];
                                for (int i = 0; i < y_s4; i++)
                                    for (int j = 0; j < x_s4; j++)
                                    {
                                        if (pixel_array1_s3[j, i] == 8)
                                        {
                                            inputs[j + i * x_s4, 0] = 0.99;
                                            st_array1[c_s3, i] = st_array1[c_s3, i] + "0";
                                        }
                                        else
                                        {
                                            inputs[j + i * x_s4, 0] = 0.01;
                                            st_array1[c_s3, i] = st_array1[c_s3, i] + " ";
                                        }

                                    }
                                double[,] targets_2 = Intelect2[0].query(inputs);
                                if (targets_2[0, 0] > 0.5)
                                {
                                    int x_s5 = 20, y_s5 = 20, num_digit = 0;
                                    int[,,] dev_digit_array = new int[1, 1, 1];
                                    Perimeter(x1_s2, h1_s3, ref pixel_array2_s3, 3, ref dev_digit_array, ref num_digit, 20);
                                    string acc_st = "";
                                    for (int d = 0; d < num_digit; d++)
                                    {
                                        if (dev_digit_array[d, 0, 0] != -2)
                                        {
                                            double[,] inputs_s5 = new double[x_s5 * y_s5, 1];
                                            for (int i = 0; i < y_s5; i++)
                                                for (int j = 0; j < x_s5; j++)
                                                    if (dev_digit_array[d, j, i] == 8)
                                                        inputs_s5[j + i * x_s5, 0] = 0.99;
                                                    else
                                                        inputs_s5[j + i * x_s5, 0] = 0.01;

                                            double[,] targets = Intelect2[1].query(inputs_s5);
                                            double max = 0;
                                            int num_max = 0;
                                            for (int i = 0; i < 11; i++)
                                                if (max < targets[i, 0])
                                                {
                                                    max = targets[i, 0];
                                                    num_max = i;
                                                }
                                            if (acc_st == "11")
                                                acc_st = "1";
                                            if (num_max == 10)
                                            {
                                                acc_st = acc_st + "А";
                                                continue;
                                            }
                                            acc_st = acc_st + num_max;
                                        }
                                        else
                                            acc_st = acc_st + "-";
                                    }
                                    st_array1[c_s3, 10] = acc_st;
                                }
                                else
                                {
                                    latitude_array1_s3[c_s3] = 0;
                                }


                            }
                        }
                        //
                        int h2_s3 = 35;
                        int[,,] lines_array2_s3 = new int[25, x2_s2, h2_s3];
                        int[] latitude_array2_s3 = new int[25];
                        string[,] st_array2 = new string[25, 21];
                        cut_lines(x2_s2, y_s1, pixel_array2_s2, 0.01, lines_array2_s3, h2_s3, 0.33, 4, 40, latitude_array2_s3);
                        for (int c_s3 = 0; c_s3 < 25; c_s3++)
                        {
                            if (lines_array2_s3[c_s3, 0, 0] != 0)
                            {
                                int x_copy = x2_s2, y_copy = h2_s3;
                                int[,] pixel_array2_s3 = new int[x_copy, y_copy];
                                for (int i = 0; i < y_copy; i++)
                                    for (int j = 0; j < x_copy; j++)
                                        pixel_array2_s3[j, i] = lines_array2_s3[c_s3, j, i];

                                delete_quadrilaterals(ref x_copy, ref y_copy, ref pixel_array2_s3, 1);
                                if (x_copy > x2_s2 || y_copy > h2_s3)
                                    cut_jpg(ref x_copy, ref y_copy, h2_s3, x2_s2, ref pixel_array2_s3);
                                if (x_copy < x2_s2 || y_copy < h2_s3)
                                    add_pixel(ref x_copy, ref y_copy, h2_s3, x2_s2, ref pixel_array2_s3);

                                int x_s4 = 100, y_s4 = 20;
                                int[,] pixel_array_s4 = new int[x_copy, y_copy];
                                for (int i = 0; i < y_copy; i++)
                                    for (int j = 0; j < x_copy; j++)
                                        pixel_array_s4[j, i] = pixel_array2_s3[j, i];
                                cut_jpg(ref x_copy, ref y_copy, y_s4, x_s4, ref pixel_array_s4);

                                //
                                double[,] inputs = new double[x_s4 * y_s4, 1];
                                for (int i = 0; i < y_s4; i++)
                                    for (int j = 0; j < x_s4; j++)
                                    {
                                        if (pixel_array_s4[j, i] == 8)
                                        {
                                            inputs[j + i * x_s4, 0] = 0.99;
                                            st_array2[c_s3, i] = st_array2[c_s3, i] + "0";
                                        }
                                        else
                                        {
                                            inputs[j + i * x_s4, 0] = 0.01;
                                            st_array2[c_s3, i] = st_array2[c_s3, i] + " ";
                                        }
                                    }
                                double[,] targets_2 = Intelect2[2].query(inputs);
                                if (targets_2[0, 0] > 0.7)
                                {
                                    ///
                                    int x_s5 = 20, y_s5 = 20, num_digit = 0;
                                    int[,,] dev_digit_array = new int[1, 1, 1];
                                    x_copy = x2_s2;
                                    y_copy = h2_s3;
                                    cut_empty(ref x_copy, ref y_copy, 0, 0, ref pixel_array2_s3, 0.01, 0.01, true);
                                    Perimeter(x_copy, y_copy, ref pixel_array2_s3, 2, ref dev_digit_array, ref num_digit, 20);
                                    string acc_st = "";
                                    for (int d = 0; d < num_digit; d++)
                                    {
                                        double[,] inputs_s5 = new double[x_s5 * y_s5, 1];
                                        for (int i = 0; i < y_s5; i++)
                                            for (int j = 0; j < x_s5; j++)
                                                if (dev_digit_array[d, j, i] == 8)
                                                    inputs_s5[j + i * x_s5, 0] = 0.99;
                                                else
                                                    inputs_s5[j + i * x_s5, 0] = 0.01;

                                        double[,] targets = Intelect2[3].query(inputs_s5);
                                        double max = 0;
                                        int num_max = 0;
                                        for (int i = 0; i < 10; i++)
                                            if (max < targets[i, 0])
                                            {
                                                max = targets[i, 0];
                                                num_max = i;
                                            }
                                        acc_st = acc_st + num_max;
                                    }
                                    st_array2[c_s3, 20] = acc_st;
                                }
                                else
                                {
                                    latitude_array2_s3[c_s3] = 0;
                                }
                            }
                        }

                        for (int i = 0; i < 65; i++)
                        {
                            if (latitude_array1_s3[i] == 0)
                                continue;
                            for (int j = 0; j < 25; j++)
                            {
                                if (latitude_array2_s3[j] == 0)
                                    continue;
                                if (Math.Abs(latitude_array1_s3[i] - latitude_array2_s3[j]) < 35)
                                {
                                    using (StreamWriter sw = new StreamWriter(@"C:\projects\OperatorVision\output.txt", true, System.Text.Encoding.Default))
                                    {
                                        for (int q = 0; q < 11; q++)
                                            sw.WriteLine(st_array1[i, q]);
                                        for (int q = 0; q < 21; q++)
                                            sw.WriteLine(st_array2[j, q]);
                                    }
                                }

                            }
                        }


                    }
                }
                await Task.Delay(1000);
            }
            ///
            ///

            ////////cut_image empty_lines
            //for (int ph = 0; ph < photo_address.Length; ph++)
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = originalImage.Width;
            //        int y = originalImage.Height;
            //        int[,] pixel_array = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        cut_empty(ref x, ref y, 0, 0, ref pixel_array, 0.1, 0.1, true);
            //        if (x != 0 && y != 0)
            //        {
            //            Bitmap result = new Bitmap(x, y);
            //            for (int i = 0; i < y; i++)
            //                for (int j = 0; j < x; j++)
            //                {
            //                    if (pixel_array[j, i] == 8)
            //                        result.SetPixel(j, i, System.Drawing.Color.Black);
            //                    else
            //                        result.SetPixel(j, i, System.Drawing.Color.White);
            //                }
            //            result.Save(@"C:\projects\OperatorVision\OperatorVision\images\scan\" + (142 + ph) + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //            result.Dispose();
            //        }
            //    }
            //    await Task.Delay(1000);
            //}

            ///////cut_image
            //for (int ph = 0; ph < photo_address.Length; ph++)
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = 185, x2 = 370, y = originalImage.Height;
            //        Bitmap btmImage = (Bitmap)originalImage;
            //        Bitmap result = new Bitmap(x, y);
            //        Bitmap result2 = new Bitmap(x2, y);
            //        for (int h = 0; h < y; h++)
            //        {
            //            for (int w = 0; w < x; w++)
            //                result.SetPixel(w, h, btmImage.GetPixel(w + 5, h));
            //            for (int w = 0; w < x2; w++)
            //                result2.SetPixel(w, h, btmImage.GetPixel(w + 1700, h)); 
            //        }
            //        result.Save(@"C:\projects\OperatorVision\OperatorVision\images\acc_area\" + ph + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //        result2.Save(@"C:\projects\OperatorVision\OperatorVision\images\writing_area\" + ph + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //        result.Dispose();
            //        result2.Dispose();
            //        await Task.Delay(100);
            //    }
            //    await Task.Delay(1000);
            //}

            ///////cut_lines
            //for (int ph = 0; ph < photo_address.Length; ph++)
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = originalImage.Width, y = originalImage.Height, h = 35; // h = 20
            //        int[,] pixel_array = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        int[,,] lines_array = new int[65, x, h];
            //        cut_lines(x, y, pixel_array, 0.01, lines_array, h, 0.33, 3, 40); // if h==20 then kef =1     0.05 // br_lim ==3 line_length_lim == 10
            //        for (int c = 0; c < 65; c++)
            //        {
            //            if (lines_array[c, 0, 0] != 0)
            //            {
            //                Bitmap result = new Bitmap(x, h);
            //                for (int i = 0; i < h; i++)
            //                    for (int j = 0; j < x; j++)
            //                    {
            //                        if (lines_array[c, j, i] == 8)
            //                            result.SetPixel(j, i, System.Drawing.Color.Black);
            //                        if (lines_array[c, j, i] == 4)
            //                            result.SetPixel(j, i, System.Drawing.Color.Red);
            //                        if (lines_array[c, j, i] == 1)
            //                            result.SetPixel(j, i, System.Drawing.Color.White);
            //                    }
            //                result.Save(@"C:\projects\OperatorVision\OperatorVision\images\wr_area_lines\" + ph + "_" + c + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //                result.Dispose();
            //            }
            //        }
            //    }
            //    await Task.Delay(100);
            //}

            //int width = 100, height = 10;
            //int[,,] output_array = new int[photo_address.Length, width, height];
            //int[,,] output_array2 = new int[photo_address.Length, 185, 20];
            //for (int ph = 0; ph < photo_address.Length; ph++)
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = originalImage.Width, x_copy = originalImage.Width;
            //        int y = originalImage.Height, y_copy = originalImage.Height;
            //        int[,] pixel_array = new int[x, y];
            //        int[,] pixel_array_copy = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //                pixel_array_copy[j, i] = pixel_array[j, i];
            //        cut_jpg(ref x, ref y, height, width, ref pixel_array);
            //        if (x != width || y != height)
            //        {
            //            output_array[ph, 0, 0] = -1;
            //        }
            //        else
            //        {
            //            for (int i = 0; i < y; i++)
            //                for (int j = 0; j < x; j++)
            //                    output_array[ph, j, i] = pixel_array[j, i];

            //            for (int i = 0; i < y_copy; i++)
            //                for (int j = 0; j < x_copy; j++)
            //                    output_array2[ph, j, i] = pixel_array_copy[j, i];

            //        }
            //    }
            //    await Task.Delay(100);
            //}
            //save_input(photo_address.Length, output_array, height, width, @"C:\projects\OperatorVision\OperatorVision\txt\output_data_7_0.txt");
            //save_input(photo_address.Length, output_array2, 20, 185, @"C:\projects\OperatorVision\OperatorVision\txt\input_data_7_0_0.txt");


            ///



            ///  delete quadrilaterals
            //for (int ph = 0; ph < photo_address.Length; ph++) 
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = originalImage.Width;
            //        int y = originalImage.Height;
            //        int x_copy = x, y_copy = y;
            //        int[,] pixel_array = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        delete_quadrilaterals(ref x_copy, ref y_copy, ref pixel_array, 1);

            //        if (x_copy > x || y_copy > y)
            //            cut_jpg(ref x_copy, ref y_copy, y, x, ref pixel_array);
            //        if (x_copy < x || y_copy < y)
            //            add_pixel(ref x_copy, ref y_copy, y, x, ref pixel_array);


            //        // save
            //        Bitmap result = new Bitmap(x, y);
            //        for (int i = 0; i < y; i++)
            //        {
            //            for (int j = 0; j < x; j++)
            //            {
            //                if (pixel_array[j, i] == 8)
            //                {
            //                    result.SetPixel(j, i, System.Drawing.Color.Black);
            //                    continue;
            //                }
            //                result.SetPixel(j, i, System.Drawing.Color.White);
            //            }
            //        }
            //        char[] char_arr = photo_address[ph].ToCharArray();
            //        string m = "";
            //        for (int i = char_arr.Length - 1; i > -1; i--)
            //            if (char_arr[i] == '\\')
            //            {
            //                char[] marked = new char[1];
            //                marked[0] = char_arr[i + 1];
            //                if (marked[0] == 'm')
            //                    m = "m";
            //                break;
            //            }
            //        result.Save(@"C:\projects\OperatorVision\OperatorVision\images\wr_no_quad\" + m + ph + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //        result.Dispose();
            //    }
            //    await Task.Delay(100);
            //}
            ///

            //int width = 100, height = 20;
            //int[,,] output_array = new int[photo_address.Length, width, height];
            //int[,,] output_array2 = new int[photo_address.Length, 370, 35];
            //for (int ph = 0; ph < photo_address.Length; ph++)
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = originalImage.Width, x_copy = originalImage.Width;
            //        int y = originalImage.Height, y_copy = originalImage.Height;
            //        int[,] pixel_array = new int[x, y];
            //        int[,] pixel_array_copy = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //                pixel_array_copy[j, i] = pixel_array[j, i];
            //        cut_jpg(ref x, ref y, height, width, ref pixel_array);
            //        if (x != width || y != height)
            //        {
            //            output_array[ph, 0, 0] = -1;
            //            output_array2[ph, 0, 0] = -1;
            //        }
            //        else
            //        {
            //            for (int i = 0; i < y; i++)
            //                for (int j = 0; j < x; j++)
            //                    output_array[ph, j, i] = pixel_array[j, i];

            //            ///
            //            char[] char_arr = photo_address[ph].ToCharArray();
            //            for (int i = char_arr.Length - 1; i > -1; i--)
            //                if (char_arr[i] == '\\')
            //                {
            //                    char[] marked = new char[1];
            //                    marked[0] = char_arr[i + 1];
            //                    if (marked[0] == 'm')
            //                        output_array[ph, 0, 0] = 10;
            //                    break;
            //                }
            //            ///
            //            for (int i = 0; i < y_copy; i++)
            //                for (int j = 0; j < x_copy; j++)
            //                    output_array2[ph, j, i] = pixel_array_copy[j, i];

            //        }
            //    }
            //    await Task.Delay(100);
            //}
            //save_input(photo_address.Length, output_array, height, width, @"C:\projects\OperatorVision\OperatorVision\txt\output_data_7_1.txt");
            //save_input(photo_address.Length, output_array2, 35, 370, @"C:\projects\OperatorVision\OperatorVision\txt\input_data_7_1_0.txt");

            //



            //// first part
            //int x = 100, y = 10, num = 0;
            //int mult = x * y;
            //int[,,] output_array = new int[1, 1, 1];
            //int[,,] input_array = new int[1, 1, 1];
            //read_input(ref output_array, @"C:\projects\OperatorVision\OperatorVision\txt\output_data_7_0.txt", y, x, ref num);
            //read_input(ref input_array, @"C:\projects\OperatorVision\OperatorVision\txt\input_data_7_0_0.txt", 20, 185, ref num);
            //num = num - 200;

            ///////
            //for (int d1 = 0; d1 < 15; d1++)
            //{
            //    double delta = 0.01 + 0.0025 * d1;
            //    Intelect2[0] = new neural_network_2hid(mult, 200, 75, 1, delta, @"C:\projects\OperatorVision\OperatorVision\txt\weight_2_0.txt"); // null
            //    //for (int c = 0; c < 10000; c++)
            //    //{
            //    //    Random rnd = new Random();
            //    //    double rand1 = Convert.ToInt32((num / 2) + (rnd.NextDouble() - 0.5) * num);
            //    //    int r = Convert.ToInt32(rand1);

            //    //    double[,] inputs = new double[mult, 1];
            //    //    for (int i = 0; i < y; i++)
            //    //        for (int j = 0; j < x; j++)
            //    //        {
            //    //            if (output_array[r, j, i] == 8)
            //    //                inputs[j + i * x, 0] = 0.99;
            //    //            else
            //    //                inputs[j + i * x, 0] = 0.01;
            //    //        }
            //    //    bool find = false;
            //    //    for (int i = 0; i < y; i++)
            //    //        for (int j = 0; j < x; j++)
            //    //            if (output_array[r, j, i] == 4)
            //    //            {
            //    //                find = true;
            //    //                break;
            //    //            }

            //    //    double[,] targets_2 = new double[1, 1];
            //    //    if (find)
            //    //    {
            //    //        targets_2[0, 0] = 0.99;
            //    //        Intelect2[0].train(inputs, targets_2);
            //    //    }
            //    //    else
            //    //    {
            //    //        targets_2[0, 0] = 0.01;
            //    //        Intelect2[0].train(inputs, targets_2);
            //    //    }
            //    //}

            //    int error_1 = 0;
            //    for (int c = num; c < num + 200; c++) // 500
            //    {
            //        bool find = false;
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //                if (output_array[c, j, i] == 4)
            //                {
            //                    find = true;
            //                    break;
            //                }
            //        //
            //        double[,] inputs = new double[mult, 1];
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //            {
            //                if (output_array[c, j, i] == 8)
            //                    inputs[j + i * x, 0] = 0.99;
            //                else
            //                    inputs[j + i * x, 0] = 0.01;
            //            } //   System.Diagnostics.Debug.WriteLine("_____________");
            //        double[,] targets_2 = Intelect2[0].query(inputs);
            //        if (targets_2[0, 0] > 0.5)
            //        {
            //            if (!find)
            //                error_1++;
            //            Bitmap result = new Bitmap(185, 20);
            //            for (int i = 0; i < 20; i++)
            //            {
            //                //string st = "";
            //                for (int j = 0; j < 185; j++)
            //                {
            //                    if (input_array[c, j, i] == 8)
            //                    {
            //                        result.SetPixel(j, i, System.Drawing.Color.Black);
            //                        //   st = st + 0;
            //                        continue;

            //                    }
            //                    result.SetPixel(j, i, System.Drawing.Color.White);
            //                    //    st = st + " ";
            //                }
            //                //   System.Diagnostics.Debug.WriteLine(st);
            //            }
            //            result.Save(@"C:\projects\OperatorVision\OperatorVision\images\test1\" + c + "_" + targets_2[0, 0] + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //            result.Dispose();
            //            await Task.Delay(100);
            //            //  System.Diagnostics.Debug.WriteLine("_____________");
            //        }
            //        else
            //        {
            //            if (find)
            //            {
            //                error_1++;
            //                Bitmap result = new Bitmap(185, 20);
            //                for (int i = 0; i < 20; i++)
            //                {
            //                    //string st = "";
            //                    for (int j = 0; j < 185; j++)
            //                    {
            //                        if (input_array[c, i, j] == 8)
            //                        {
            //                            result.SetPixel(j, i, System.Drawing.Color.Black);
            //                            //   st = st + 0;
            //                            continue;

            //                        }
            //                        result.SetPixel(j, i, System.Drawing.Color.White);
            //                        //    st = st + " ";
            //                    }
            //                    //   System.Diagnostics.Debug.WriteLine(st);
            //                }
            //                result.Save(@"C:\projects\OperatorVision\OperatorVision\images\test1\" + c + "_" + targets_2[0, 0] + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //                result.Dispose();
            //                await Task.Delay(100);
            //            }

            //            //System.Diagnostics.Debug.WriteLine("!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            //            //System.Diagnostics.Debug.WriteLine("_____________");
            //        }

            //    }
            //    //using (StreamWriter sw = new StreamWriter(@"C:\projects\OperatorVision\OperatorVision\txt\test.txt", true, System.Text.Encoding.Default))
            //    //{
            //    //    sw.WriteLine("kef: " + delta + "\t" + "err_1: " + error_1);
            //    //}
            //    //Intelect2[0].save(@"C:\projects\OperatorVision\OperatorVision\txt\weight_2_0.txt");
            //}


            /// mechanic digits
            //int width = 20, height = 20;
            //int[,,] output_array = new int[photo_address.Length, width, height];
            //for (int ph = 0; ph < photo_address.Length; ph++)
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        int x = originalImage.Width, x_copy = originalImage.Width;
            //        int y = originalImage.Height, y_copy = originalImage.Height;
            //        int[,] pixel_array = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        if (x != width || y != height)
            //        {
            //            output_array[ph, 0, 0] = -1;
            //        }
            //        else
            //        {
            //            for (int i = 0; i < y; i++)
            //                for (int j = 0; j < x; j++)
            //                    output_array[ph, j, i] = pixel_array[j, i];

            //        }
            //    }
            //    await Task.Delay(100);
            //}
            //save_input(photo_address.Length, output_array, height, width, @"C:\projects\OperatorVision\OperatorVision\txt\mechanic_digits.txt");



            //int x = 20, y = 20, num = 0;  // 28 28
            //int mult = x * y;
            //int[,,] input_array = new int[1, 1, 1];
            //read_input(ref input_array, @"C:\projects\OperatorVision\OperatorVision\txt\mechanic_digits.txt", x, y, ref num);
            //num = num - 1;
            //for (int d1 = 0; d1 < 10; d1++)
            //{
            //    double delta = 0.01 + 0.005 * d1;
            //    Intelect2[0] = new neural_network_2hid(mult, 200, 75, 11, delta, null); //null or  @"C:\projects\OperatorVision\OperatorVision\txt\weight_1_0.txt"

            //    for (int c = 0; c < 10000; c++)
            //    {
            //        Random rnd = new Random();
            //        double rand1 = Convert.ToInt32((num / 2) + (rnd.NextDouble() - 0.5) * num);
            //        int r = Convert.ToInt32(rand1);

            //        double[,] inputs = new double[mult, 1];
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //            {
            //                if (input_array[r, j, i] == 8)
            //                    inputs[j + i * x, 0] = 0.99;
            //                else
            //                    inputs[j + i * x, 0] = 0.01;
            //            }
            //        double[,] targets = new double[11, 1];
            //        for (int i = 0; i < 11; i++)
            //            targets[i, 0] = 0.01;
            //        int dig = r / 50;
            //        targets[dig, 0] = 0.99;
            //        Intelect2[0].train(inputs, targets);
            //    }

            //    ///
            //    //int[,,] dev_digit_array = new int[1, 1, 1];
            //    //int num_digit = 0;
            //    //for (int ph = 0; ph < photo_address.Length; ph++)  //
            //    //{
            //    //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    //    {
            //    //        int x1 = originalImage.Width;
            //    //        int y1 = originalImage.Height;
            //    //        int[,] pixel_array = new int[x1, y1];
            //    //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //    //        Perimeter(x1, y1, ref pixel_array, 3, ref dev_digit_array, ref num_digit, 20);

            //    //        //for (int d = 0; d < num_digit; d++)
            //    //        //{
            //    //        //    if (dev_digit_array[d, 0, 0] != -2)
            //    //        //    {
            //    //        //        double[,] inputs = new double[mult, 1];
            //    //        //        Bitmap result = new Bitmap(x, y);
            //    //        //        for (int i = 0; i < y; i++)
            //    //        //        {
            //    //        //            string st = "";
            //    //        //            for (int j = 0; j < x; j++)
            //    //        //            {

            //    //        //                if (dev_digit_array[d, j, i] == 8)
            //    //        //                {
            //    //        //                    result.SetPixel(j, i, System.Drawing.Color.Black);
            //    //        //                    inputs[j + i * x, 0] = 0.99;
            //    //        //                    st = st + 0;
            //    //        //                }
            //    //        //                else
            //    //        //                {
            //    //        //                    result.SetPixel(j, i, System.Drawing.Color.White);
            //    //        //                    inputs[j + i * x, 0] = 0.01;
            //    //        //                    st = st + " ";
            //    //        //                }
            //    //        //            }
            //    //        //            System.Diagnostics.Debug.WriteLine(st);
            //    //        //        }

            //    //        //        double[,] targets = Intelect2[0].query(inputs);
            //    //        //        double max = 0;
            //    //        //        double num_max = 0;
            //    //        //        string st1 = "";
            //    //        //        for (int i = 0; i < 10; i++)
            //    //        //        {
            //    //        //            st1 = st1 + targets[i, 0] + "  ";
            //    //        //            if (max < targets[i, 0])
            //    //        //            {
            //    //        //                max = targets[i, 0];
            //    //        //                num_max = i;
            //    //        //            }
            //    //        //        }
            //    //        //        System.Diagnostics.Debug.WriteLine(num_max);
            //    //        //        System.Diagnostics.Debug.WriteLine(st1);
            //    //        //        result.Save(@"C:\projects\OperatorVision\OperatorVision\images\n_network_mech_digits\" + num_max + "_" + max + "_" + ph + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //    //        //        result.Dispose();
            //    //        //    }
            //    //        //}

            //    //        //string acc_st = "";

            //    //        //for (int d = 0; d < num_digit; d++)
            //    //        //{
            //    //        //    if (dev_digit_array[d, 0, 0] != -2)
            //    //        //    {
            //    //        //        double[,] inputs = new double[mult, 1];
            //    //        //        for (int i = 0; i < y; i++)
            //    //        //            for (int j = 0; j < x; j++)
            //    //        //                if (dev_digit_array[d, j, i] == 8)
            //    //        //                    inputs[j + i * x, 0] = 0.99;
            //    //        //                else
            //    //        //                    inputs[j + i * x, 0] = 0.01;

            //    //        //        double[,] targets = Intelect2[0].query(inputs);
            //    //        //        double max = 0;
            //    //        //        double num_max = 0;
            //    //        //        string st1 = "";
            //    //        //        for (int i = 0; i < 10; i++)
            //    //        //        {
            //    //        //            st1 = st1 + targets[i, 0] + "  ";
            //    //        //            if (max < targets[i, 0])
            //    //        //            {
            //    //        //                max = targets[i, 0];
            //    //        //                num_max = i;
            //    //        //            }
            //    //        //        }
            //    //        //        acc_st = acc_st + num_max;
            //    //        //    }
            //    //        //    else
            //    //        //        acc_st = acc_st + "-";
            //    //        //}
            //    //    }
            //    //    await Task.Delay(200);
            //    //}

            //    Intelect2[0].save(@"C:\projects\OperatorVision\OperatorVision\txt\weight_1_0.txt");
            //}

            ///
            ///// second part
            //int x = 100, y = 20, num = 0;
            //int mult = x * y;
            //int[,,] output_array = new int[1, 1, 1];
            //int[,,] input_array = new int[1, 1, 1];
            //read_input(ref output_array, @"C:\projects\OperatorVision\OperatorVision\txt\output_data_7_1.txt", y, x, ref num);
            //read_input(ref input_array, @"C:\projects\OperatorVision\OperatorVision\txt\input_data_7_1_0.txt", 35, 370, ref num);
            //num = num - 50;

            ///////
            //for (int d1 = 0; d1 < 15; d1++)
            //{
            //    double delta = 0.01 + 0.0025 * d1;
            //    Intelect2[0] = new neural_network_2hid(mult, 200, 75, 1, delta, @"C:\projects\OperatorVision\OperatorVision\txt\weight_3_0.txt" );  //null
            //    //for (int c = 0; c < 10000; c++)
            //    //{
            //    //    Random rnd = new Random();
            //    //    double rand1 = Convert.ToInt32((num / 2) + (rnd.NextDouble() - 0.5) * num);
            //    //    int r = Convert.ToInt32(rand1);

            //    //    double[,] inputs = new double[mult, 1];
            //    //    for (int i = 0; i < y; i++)
            //    //        for (int j = 0; j < x; j++)
            //    //        {
            //    //            if (output_array[r, j, i] == 8)
            //    //                inputs[j + i * x, 0] = 0.99;
            //    //            else
            //    //                inputs[j + i * x, 0] = 0.01;
            //    //        }
            //    //    bool find = true;
            //    //    if (output_array[r, 0, 0] == -1)
            //    //        find = false;
            //    //    double[,] targets_2 = new double[1, 1];
            //    //    if (find)
            //    //    {
            //    //        targets_2[0, 0] = 0.99;
            //    //        Intelect2[0].train(inputs, targets_2);
            //    //    }
            //    //    else
            //    //    {
            //    //        targets_2[0, 0] = 0.01;
            //    //        Intelect2[0].train(inputs, targets_2);
            //    //    }
            //    //}

            //    int error_1 = 0;
            //    for (int c = num; c < num + 50; c++) // 500
            //    {
            //        bool find = true;
            //        if (output_array[c, 0, 0] == -1)
            //            find = false;
            //        //
            //        double[,] inputs = new double[mult, 1];
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //            {
            //                if (output_array[c, j, i] == 8)
            //                    inputs[j + i * x, 0] = 0.99;
            //                else
            //                    inputs[j + i * x, 0] = 0.01;
            //            } //   System.Diagnostics.Debug.WriteLine("_____________");
            //        double[,] targets_2 = Intelect2[0].query(inputs);
            //        if (targets_2[0, 0] > 0.7)
            //        {
            //            if (!find)
            //                error_1++;
            //            Bitmap result = new Bitmap(370, 35);
            //            for (int i = 0; i < 35; i++)
            //            {
            //                //string st = "";
            //                for (int j = 0; j < 370; j++)
            //                {
            //                    if (input_array[c, j, i] == 8)
            //                    {
            //                        result.SetPixel(j, i, System.Drawing.Color.Black);
            //                        //   st = st + 0;
            //                        continue;

            //                    }
            //                    result.SetPixel(j, i, System.Drawing.Color.White);
            //                    //    st = st + " ";
            //                }
            //                //   System.Diagnostics.Debug.WriteLine(st);
            //            }
            //            result.Save(@"C:\projects\OperatorVision\OperatorVision\images\test2\" + c + "_" + targets_2[0, 0] + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //            result.Dispose();
            //            await Task.Delay(100);
            //            //  System.Diagnostics.Debug.WriteLine("_____________");
            //        }
            //        else
            //        {
            //            if (find)
            //            {
            //                error_1++;
            //                Bitmap result = new Bitmap(370, 35);
            //                for (int i = 0; i < 35; i++)
            //                {
            //                    //string st = "";
            //                    for (int j = 0; j < 370; j++)
            //                    {
            //                        if (input_array[c, j, i] == 8)
            //                        {
            //                            result.SetPixel(j, i, System.Drawing.Color.Black);
            //                            //   st = st + 0;
            //                            continue;

            //                        }
            //                        result.SetPixel(j, i, System.Drawing.Color.White);
            //                        //    st = st + " ";
            //                    }
            //                    //   System.Diagnostics.Debug.WriteLine(st);
            //                }
            //                result.Save(@"C:\projects\OperatorVision\OperatorVision\images\test2\" + c + "_" + targets_2[0, 0] + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //                result.Dispose();
            //                await Task.Delay(100);
            //            }

            //            //System.Diagnostics.Debug.WriteLine("!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            //            //System.Diagnostics.Debug.WriteLine("_____________");
            //        }

            //    }
            //    using (StreamWriter sw = new StreamWriter(@"C:\projects\OperatorVision\OperatorVision\txt\test.txt", true, System.Text.Encoding.Default))
            //    {
            //        sw.WriteLine("kef: " + delta + "\t" + "err_1: " + error_1);
            //    }
            //  //  Intelect2[0].save(@"C:\projects\OperatorVision\OperatorVision\txt\weight_3_0.txt");
            //}


            /// third part
            //int x = 28, y = 28;
            //int[] digit_array = new int[photo_address.Length];
            //int[,,] output_array = new int[photo_address.Length, 28, 28];
            //for (int ph = 0; ph < photo_address.Length; ph++)  //
            //{
            //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    {
            //        char[] char_arr = photo_address[ph].ToCharArray();
            //        for (int i = char_arr.Length - 1; i > -1; i--)
            //            if (char_arr[i] == '\\')
            //            {
            //                char[] digit = new char[1];
            //                digit[0] = char_arr[i + 1];
            //                digit_array[ph] = Convert.ToInt16(new string(digit));
            //                break;
            //            }
            //        int[,] pixel_array = new int[x, y];
            //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //        int x_copy = x, y_copy = y;
            //        cut_jpg(ref x_copy, ref y_copy, 20, 20, ref pixel_array);
            //        for (int y1 = 0; y1 < y_copy; y1++)
            //            for (int x1 = 0; x1 < x_copy; x1++)
            //                output_array[ph, x1, y1] = pixel_array[x1, y1];
            //    }
            //    await Task.Delay(25);
            //}
            //save_input(photo_address.Length, output_array, 20, 20, @"C:\projects\OperatorVision\OperatorVision\txt\controller_digits3.txt");
            //using (StreamWriter sw = new StreamWriter(@"C:\projects\OperatorVision\OperatorVision\txt\digits4.txt", true, System.Text.Encoding.Default))
            //{
            //    for (int ph = 0; ph < photo_address.Length; ph++)
            //        sw.WriteLine(digit_array[ph]);
            //}




            /// 
            //int x = 20, y = 20, num = 0;  // 28 28
            //int mult = x * y;
            //int[,,] input_array = new int[1, 1, 1];
            //int[] digit_array = new int[1];
            //using (StreamReader sr1 = new StreamReader(@"C:\projects\OperatorVision\OperatorVision\txt\digits4.txt"))
            //{
            //    List<string> all_lines = new List<string>();
            //    string line;
            //    while ((line = sr1.ReadLine()) != null)
            //        all_lines.Add(line);
            //    digit_array = new int[all_lines.Count];
            //    for (int i = 0; i < all_lines.Count; i++)
            //        digit_array[i] = Convert.ToInt16(all_lines[i]);
            //}
            //read_input(ref input_array, @"C:\projects\OperatorVision\OperatorVision\txt\controller_digits3.txt", x, y, ref num);
            //num = num - 100;
            //for (int d1 = 0; d1 < 10; d1++)
            //{
            //    double delta = 0.01 + 0.005 * d1;
            //    Intelect2[0] = new neural_network_2hid(mult, 200, 75, 10, delta, null); //null or  @"C:\projects\OperatorVision\OperatorVision\txt\weight_4_0.txt"

            //    for (int c = 0; c < 100000; c++)
            //    {
            //        Random rnd = new Random();
            //        double rand1 = Convert.ToInt32((num / 2) + (rnd.NextDouble() - 0.5) * num);
            //        int r = Convert.ToInt32(rand1);

            //        double[,] inputs = new double[mult, 1];
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //            {
            //                if (input_array[r, j, i] == 8)
            //                    inputs[j + i * x, 0] = 0.99;
            //                else
            //                    inputs[j + i * x, 0] = 0.01;
            //            }
            //        double[,] targets = new double[10, 1];
            //        for (int i = 0; i < 10; i++)
            //            targets[i, 0] = 0.01;
            //        targets[digit_array[r], 0] = 0.99;
            //        Intelect2[0].train(inputs, targets);
            //    }

            //    // check
            //    int success_check = 0;
            //    for (int c = num; c < num + 100; c++)
            //    {
            //        double[,] inputs = new double[mult, 1];
            //        for (int i = 0; i < y; i++)
            //            for (int j = 0; j < x; j++)
            //            {
            //                if (input_array[c, j, i] == 8)
            //                    inputs[j + i * x, 0] = 0.99;
            //                else
            //                    inputs[j + i * x, 0] = 0.01;
            //            }

            //        double[,] targets = Intelect2[0].query(inputs);
            //        double max = 0;
            //        int num_max = 0;
            //        for (int i = 0; i < 10; i++)
            //            if (max < targets[i, 0])
            //            {
            //                max = targets[i, 0];
            //                num_max = i;
            //            }
            //        if (num_max == digit_array[c])
            //            success_check++;
            //    }
            //    double dev = Convert.ToDouble(success_check) / 100;
            //    using (StreamWriter sw = new StreamWriter(@"C:\projects\OperatorVision\OperatorVision\txt\test.txt", true, System.Text.Encoding.Default))
            //    {
            //        sw.WriteLine("kef: " + delta + "\t" + "percent: " + dev);
            //    }

            //    ///
            //    //int[,,] dev_digit_array = new int[1, 1, 1];
            //    //int num_digit = 0;
            //    //for (int ph = 0; ph < photo_address.Length; ph++)  //
            //    //{
            //    //    using (Image originalImage = Image.FromFile(photo_address[ph]))
            //    //    {
            //    //        int x1 = originalImage.Width;
            //    //        int y1 = originalImage.Height;
            //    //        int[,] pixel_array = new int[x1, y1];
            //    //        Int_palette((Bitmap)originalImage, 2, pixel_array);
            //    //        cut_empty(ref x1, ref y1, 0, 0, ref pixel_array, 0.05, 0.05, true);
            //    //        Perimeter(x1, y1, ref pixel_array, 2, ref dev_digit_array, ref num_digit);

            //    //        for (int d = 0; d < num_digit; d++)
            //    //        {
            //    //            double[,] inputs = new double[mult, 1];
            //    //            Bitmap result = new Bitmap(x, y);
            //    //            for (int i = 0; i < y; i++)
            //    //            {
            //    //                string st = "";
            //    //                for (int j = 0; j < x; j++)
            //    //                {

            //    //                    if (dev_digit_array[d, j, i] == 8)
            //    //                    {
            //    //                        result.SetPixel(j, i, System.Drawing.Color.Black);
            //    //                        inputs[j + i * x, 0] = 0.99;
            //    //                        st = st + 0;
            //    //                    }
            //    //                    else
            //    //                    {
            //    //                        result.SetPixel(j, i, System.Drawing.Color.White);
            //    //                        inputs[j + i * x, 0] = 0.01;
            //    //                        st = st + " ";
            //    //                    }
            //    //                }
            //    //                System.Diagnostics.Debug.WriteLine(st);
            //    //            }

            //    //            double[,] targets = Intelect2[0].query(inputs);
            //    //            double max = 0;
            //    //            double num_max = 0;
            //    //            string st1 = "";
            //    //            for (int i = 0; i < 10; i++)
            //    //            {
            //    //                st1 = st1 + targets[i, 0] + "  ";
            //    //                if (max < targets[i, 0])
            //    //                {
            //    //                    max = targets[i, 0];
            //    //                    num_max = i;
            //    //                }
            //    //            }
            //    //            System.Diagnostics.Debug.WriteLine(num_max);
            //    //            System.Diagnostics.Debug.WriteLine(st1);
            //    //            result.Save(@"C:\projects\OperatorVision\OperatorVision\images\n_network_marked_digits2\" + num_max + "_" + max + "_" + ph + ".png", System.Drawing.Imaging.ImageFormat.Png);
            //    //            result.Dispose();
            //    //        }

            //    //    }
            //    //    await Task.Delay(200);
            //    //}
            //    Intelect2[0].save(@"C:\projects\OperatorVision\OperatorVision\txt\weight_4_0.txt");
            //}
        }
    }
}
