package Project
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import scala.io.StdIn

object MovieAnalysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Movie Analysis")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    // Load movies data
    val moviesDF = spark.read
      .option("header",value= false)
      .option("delimiter", "::")
      .csv("data/Project/movies.dat")
      .toDF("MovieID", "Title", "Genres")

    // Load ratings data1

    val ratingsDF = spark.read
      .option("header",value= false)
      .option("delimiter", "::")
      .csv("data/Project/ratings.dat")
      .toDF("UserID", "MovieID", "Rating", "Timestamp")
      .withColumn("UserID", col("UserID").cast("int"))
      .withColumn("MovieID", col("MovieID").cast("int"))
      .withColumn("Rating", col("Rating").cast("float"))


    var exit = false
    while (!exit) {
      println("\nMovie Analysis Menu:")
      println("1. Top 10 most viewed movies")
      println("2. Latest released movies")
      println("3. Number of movies starting with each letter/number")
      println("4. Number of movies per genre")
      println("5. Distinct list of genres")
      println("6. Add a movie")
      println("7. Delete a movie")
      println("8. Update a movie")
      println("9. Exit")
      println("10. Predict movie rating using ALS")
      print("Enter your choice: ")

      val choice = StdIn.readInt()

      choice match {
        case 1 =>
          val topMovies = ratingsDF.groupBy("MovieID")
            .agg(count("MovieID").alias("ViewCount"))
            .orderBy(desc("ViewCount"))
            .limit(10)
            .join(moviesDF, "MovieID")
            .select("Title", "ViewCount")
          println("Top 10 most viewed movies:")
          topMovies.show(false)

        case 2 =>
          val latestMovies = moviesDF.withColumn("Year", regexp_extract(col("Title"), "\\((\\d{4})\\)", 1))
            .filter(col("Year").isNotNull && col("Year") =!= "")
            .orderBy(desc("Year"))
            .select("Title", "Year")
            .limit(10)
          println("Latest released movies:")
          latestMovies.show(false)

        case 3 =>
          val movieStartCounts = moviesDF.withColumn("FirstChar", substring(col("Title"), 1, 1))
            .groupBy("FirstChar").count()
            .orderBy("FirstChar")
          println("Number of movies starting with each letter/number:")
          movieStartCounts.show(false)

        case 4 =>
          val genreCounts = moviesDF.withColumn("Genre", explode(split(col("Genres"), "\\|")))
            .groupBy("Genre").count()
            .orderBy(desc("count"))
          println("Number of movies per genre:")
          genreCounts.show(false)

        case 5 =>
          val distinctGenres = moviesDF.select(explode(split(col("Genres"), "\\|"))).distinct()
            .toDF("Genre")
            .orderBy("Genre")
          println("Distinct list of genres:")
          distinctGenres.show(false)

        case 6 =>
          print("Enter MovieID: ")
          val movieID = StdIn.readLine()
          print("Enter Title: ")
          val title = StdIn.readLine()
          print("Enter Genres: ")
          val genres = StdIn.readLine()
          val newMovie = Seq((movieID, title, genres)).toDF("MovieID", "Title", "Genres")
          val updatedDF = moviesDF.union(newMovie)
          println("Movie added successfully!")
          updatedDF.show(false)

        case 7 =>
          print("Enter MovieID to delete: ")
          val movieID = StdIn.readLine()
          val updatedDF = moviesDF.filter(col("MovieID") =!= movieID)
          println("Movie deleted successfully!")
          updatedDF.show(false)

        case 8 =>
          print("Enter MovieID to update: ")
          val movieID = StdIn.readLine()
          print("Enter new Title: ")
          val title = StdIn.readLine()
          print("Enter new Genres: ")
          val genres = StdIn.readLine()
          val filteredDF = moviesDF.filter(col("MovieID") =!= movieID)
          val updatedMovie = Seq((movieID, title, genres)).toDF("MovieID", "Title", "Genres")
          val updatedDF = filteredDF.union(updatedMovie)
          println("Movie updated successfully!")
          updatedDF.show(false)

        case 9 =>
          println("Exiting program...")
          exit = true

        case 10 =>
          println("Predicting movie rating using ALS...")
          val als = new ALS()
            .setMaxIter(10)
            .setRegParam(0.1)
            .setUserCol("UserID")
            .setItemCol("MovieID")
            .setRatingCol("Rating")

          val model = als.fit(ratingsDF)

          print("Enter UserID: ")
          val userID = StdIn.readInt()
          print("Enter MovieID: ")
          val movieID = StdIn.readInt()

          val testDF = Seq((userID, movieID)).toDF("UserID", "MovieID")
          val predictions = model.transform(testDF)

          predictions.select("MovieID", "prediction").show(false)

        case _ =>
          println("Invalid choice. Please try again.")
      }
    }
    spark.stop()
  }
}