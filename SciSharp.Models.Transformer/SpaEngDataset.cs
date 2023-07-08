using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class SpaEngDataset
    {
        TransformerTranslationConfig cfg;

        public SpaEngDataset()
        {
            cfg = new TransformerTranslationConfig();
        }

        public SpaEngDataset(TransformerTranslationConfig cfg)
        {
            this.cfg = cfg;
        }

        protected string DownloadData()
        {
            return keras.utils.get_file(
                fname: "spa-eng.zip",
                origin: "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
                extract: true
            );
        }

        protected List<(string, string)>[] ParseData(string filePath)
        {
            List<(string, string)> textPairs = new List<(string, string)>();

            using (StreamReader sr = new StreamReader(filePath))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    string[] parts = line.Split('\t');
                    string eng = parts[0];
                    string spa = parts[1];
                    spa = "[start] " + spa + " [end]";
                    textPairs.Add((eng, spa));
                }
            }

            Random rng = new Random();
            textPairs = textPairs.OrderBy(_ => rng.Next()).ToList();

            int numValSamples = (int)(0.15 * textPairs.Count);
            int numTrainSamples = textPairs.Count - 2 * numValSamples;

            List<(string, string)> trainPairs = textPairs.GetRange(0, numTrainSamples);
            List<(string, string)> valPairs = textPairs.GetRange(numTrainSamples, numValSamples);
            List<(string, string)> testPairs = textPairs.GetRange(numTrainSamples + numValSamples, textPairs.Count - numTrainSamples - numValSamples);

            Console.WriteLine(String.Format("{0} total pairs", textPairs.Count));
            Console.WriteLine(String.Format("{0} training pairs", trainPairs.Count));
            Console.WriteLine(String.Format("{0} validation pairs", valPairs.Count));
            Console.WriteLine(String.Format("{0} test pairs", testPairs.Count));
            return new List<(string, string)>[] { trainPairs, valPairs, testPairs };
        }

        protected void VectorizeData(
            List<(string, string)> trainPairs,
            List<(string, string)> valPairs,
            List<(string, string)> testPairs)
        {
            string stripChars = "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~¿";

            var engVectorization = keras.layers.preprocessing.TextVectorization(
                max_tokens: cfg.DatasetCfg.vocab_size,
                output_mode: "int",
                output_sequence_length: cfg.DatasetCfg.maxlen);
            var spaVectorization = keras.layers.preprocessing.TextVectorization(
                max_tokens: cfg.DatasetCfg.vocab_size,
                output_mode: "int",
                output_sequence_length: cfg.DatasetCfg.maxlen + 1,
                standardize: (inputString) =>
                {
                    string lowercase = inputString.ToString().ToLower();
                    return new Tensor(Regex.Replace(lowercase, $"[{Regex.Escape(stripChars)}]", ""));
                });
            Tensor trainEngTexts = tf.constant(trainPairs.Select(pair => pair.Item1).ToArray());
            Tensor trainSpaTexts = tf.constant(trainPairs.Select(pair => pair.Item2).ToArray());
            Tensor valEngTexts = tf.constant(valPairs.Select(pair => pair.Item1).ToArray());
            Tensor valSpaTexts = tf.constant(valPairs.Select(pair => pair.Item2).ToArray());
            engVectorization.adapt(trainEngTexts);
            spaVectorization.adapt(trainSpaTexts);

            var formatDataset = new Func<Tensors, Tensors>(delegate (Tensors inputs)
            {
                Tensor[] data = inputs.ToArray();
                var eng = engVectorization.Apply(data[0]);
                var spa = spaVectorization.Apply(data[1]);
                return new Tensors(
                    eng, //"encoder_inputs": eng
                    tf.slice(spa, new[] { 0, 0 }, new[] { spa.shape[0], - 1 }), //"decoder_inputs": spa[:, :-1]
                    tf.slice(spa, new[] { 0, 1 }, new[] { spa.shape[0], - 1 })); // "outputs": spa[:, 1:]
            });

            var makeDataset = new Func<Tensor, Tensor, IDatasetV2>(delegate (Tensor eng_texts, Tensor spa_texts)
            {
                var dataset = tf.data.Dataset.from_tensor_slices(eng_texts, spa_texts);
                dataset = dataset.batch(cfg.TrainCfg.batch_size);
                dataset = dataset.map(formatDataset);
                return dataset.shuffle(2048).prefetch(16).cache();
            });

            var train_ds = makeDataset(trainEngTexts, trainSpaTexts);
            var val_ds = makeDataset(valEngTexts, valSpaTexts);
        }

        public void GetData()
        {
            var fileDir = DownloadData();
            var filePath = Path.Combine(fileDir, "spa-eng", "spa.txt");
            List<(string, string)>[] dataset = ParseData(filePath);
            List<(string, string)> trainPairs = dataset[0], valPairs = dataset[1], testPairs = dataset[2];
            VectorizeData(trainPairs, valPairs, testPairs);
        }
    }
}
