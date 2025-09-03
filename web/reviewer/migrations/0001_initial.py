# Initial migration for Feedback
from django.db import migrations, models
class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at_epoch', models.IntegerField()),
                ('crop_filename', models.CharField(max_length=512)),
                ('predicted_label', models.CharField(blank=True, max_length=128, null=True)),
                ('distance_value', models.FloatField(blank=True, null=True)),
                ('decision', models.CharField(choices=[('approve', 'Approve'), ('unknown', 'Unknown'), ('override', 'Override')], max_length=16)),
                ('corrected_label', models.CharField(blank=True, max_length=128, null=True)),
            ],
            options={'ordering': ['-id']},
        ),
    ]
