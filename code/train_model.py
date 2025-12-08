import argparse
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def get_spark(app_name: str = "WineQualityTraining"):
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_dataset(spark, path: str):
    # CSVs are ; separated with header row
    return (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .csv(path)
    )


def build_pipeline(feature_cols, label_col="quality"):
    indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=10
    )

    return Pipeline(stages=[indexer, assembler, rf])


def main(args):
    spark = get_spark()

    print(f"Reading training data from: {args.training_path}")
    train_df = load_dataset(spark, args.training_path)

    print(f"Reading validation data from: {args.validation_path}")
    val_df = load_dataset(spark, args.validation_path)

    label_col = "quality"
    feature_cols = [c for c in train_df.columns if c != label_col]

    pipeline = build_pipeline(feature_cols, label_col)

    print("Training model...")
    model = pipeline.fit(train_df)

    print("Evaluating on validation set...")
    predictions = model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator.evaluate(predictions)
    print(f"Validation F1 score: {f1}")

    print(f"Saving model to: {args.model_output_path}")
    model.write().overwrite().save(args.model_output_path)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wine-quality model with Spark.")
    parser.add_argument("training_path", help="Path to TrainingDataset.csv (local path or S3 URI).")
    parser.add_argument("validation_path", help="Path to ValidationDataset.csv (local path or S3 URI).")
    parser.add_argument("model_output_path", help="Output path to save the trained model (local dir or S3 URI).")

    args = parser.parse_args()
    main(args)

